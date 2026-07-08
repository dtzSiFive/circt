//===- ModuleInliner.cpp - FIRRTL module inlining ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements FIRRTL module instance inlining.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Support/Debug.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TrailingObjects.h"

#define DEBUG_TYPE "firrtl-inliner"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INLINER
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

using hw::InnerRefAttr;

using InnerRefToNewNameMap = DenseMap<hw::InnerRefAttr, StringAttr>;

//===----------------------------------------------------------------------===//
// Classification and NLA Prepasses
//===----------------------------------------------------------------------===//

namespace {

struct ModInfo {
  bool hasInline : 1;
  bool hasFlatten : 1;
  bool underFlatten : 1; // ANY parent flattens this, not ALL.

  /// Will this module exist in the final result?
  bool isLive : 1;

  /// Is this module reachable in a way that isn't flattened?
  bool hasUnflattenedPath : 1;

  // Can't have default member initialization of bitfield members until C++20.
  ModInfo()
      : hasInline(false), hasFlatten(false), underFlatten(false), isLive(false),
        hasUnflattenedPath(false) {}

  void markLiveRoot() { isLive = hasUnflattenedPath = true; }
};

using ModInfoMap = DenseMap<Operation *, ModInfo>;

class ClassificationPrepass {
  CircuitOp circuit;
  InstanceGraph &instanceGraph;
  SymbolTable &symbolTable;

  ModInfoMap modInfoMap;

public:
  ClassificationPrepass(CircuitOp circuit, InstanceGraph &instanceGraph,
                        SymbolTable &symbolTable)
      : circuit(circuit), instanceGraph(instanceGraph),
        symbolTable(symbolTable) {}

  const ModInfoMap &getModInfoMap() { return modInfoMap; }

  LogicalResult run();

  void dump();
};

// TODO: s/Inst/Sym/; at least if we want to support and preserve 'old-style'.
struct SurvivingHop {
  StringAttr origMod;
  StringAttr origInst;
  StringAttr finalMod;
  StringAttr finalInst; // mutable
};

class VirtualNLA final : llvm::TrailingObjects<VirtualNLA, SurvivingHop> {
  friend TrailingObjects;

  unsigned numHops;

  VirtualNLA(unsigned id, StringAttr origSym, ArrayRef<SurvivingHop> path)
      : numHops(path.size()), id(id), origSym(origSym), realizedSym() {
    llvm::uninitialized_copy(path, getTrailingObjects());
  };

public:
  unsigned id;
  StringAttr origSym;
  StringAttr realizedSym;
  bool wasUsed = false;

  static VirtualNLA *create(llvm::BumpPtrAllocator &alloc, unsigned id,
                            StringAttr origSym, ArrayRef<SurvivingHop> path) {
    size_t size = totalSizeToAlloc<SurvivingHop>(path.size());
    auto *mem = alloc.Allocate(size, alignof(VirtualNLA));
    return new (mem) VirtualNLA(id, origSym, path);
  }

  bool isLocal() const { return numHops <= 1; }

  ArrayRef<SurvivingHop> getPath() const {
    return {getTrailingObjects(), numHops};
  }

  MutableArrayRef<SurvivingHop> getPathMutable() {
    return {getTrailingObjects(), numHops};
  }

  void dump() const {
    llvm::dbgs() << llvm::formatv("    VirtualNLA {0}: origSym @{1}", id,
                                  origSym);
    if (isLocal()) {
      llvm::dbgs() << " -> local\n";
    } else {
      llvm::dbgs() << llvm::formatv(
          ", realizedSym: {0}, hops: {1}\n",
          realizedSym ? (Twine("@") + realizedSym.str()) : "<none>", numHops);
      for (const auto &hop : getPath()) {
        llvm::dbgs() << llvm::formatv(
            "      - {0}::{1} -> {2}::{3}\n", hop.origMod,
            (hop.origInst ? hop.origInst.str() : "*"), hop.finalMod,
            (hop.finalInst ? hop.finalInst.str() : "(TBD)"));
      }
    }
  }
};

using InstancePointer =
    llvm::PointerUnion<Operation *, StringAttr /* Instance symbol name */>;

// TODO: Pick one?
// Problem is we don't want spurious inner symbols on instances,
// but also pointer to operations we might delete isn't ideal,
// nor is having to chase inner symbols just to get the target.
using CoordinateKey = std::pair<StringAttr /* Module */, InstancePointer>;
// using CoordinateKey =
//     std::pair<StringAttr /* Module */, StringAttr /*Instance*/>;

class ModNamespaces {
  DenseMap<Operation *, hw::InnerSymbolNamespace> moduleNamespaces;

public:
  hw::InnerSymbolNamespace &getNamespace(FModuleLike mod) {
    return moduleNamespaces.try_emplace(mod, mod).first->second;
  }
};

class NLAPrepass {
public:
  NLAPrepass(CircuitOp circuit, CircuitNamespace &circuitNamespace,
             SymbolTable &symbolTable, InstanceGraph &instanceGraph,
             ModNamespaces &modNamespaces, const ModInfoMap &modInfoMap)
      : circuit(circuit), circuitNamespace(circuitNamespace),
        symbolTable(symbolTable), instanceGraph(instanceGraph),
        modNamespaces(modNamespaces), modInfoMap(modInfoMap) {}

  LogicalResult run();
  void dump();

private:
  LogicalResult traceUpUntilSurviving(
      StringAttr rootModName, SmallVectorImpl<CoordinateKey> &currentPath,
      SmallVectorImpl<SmallVector<CoordinateKey>> &discoveredPaths);

  void processSinglePathContext(StringAttr origSym,
                                const SmallVectorImpl<CoordinateKey> &absPath);

  CircuitOp circuit;
  CircuitNamespace &circuitNamespace;
  SymbolTable &symbolTable;
  InstanceGraph &instanceGraph;
  ModNamespaces modNamespaces;
  const ModInfoMap &modInfoMap;

  // Stable pool allocation for virtual NLA structures.
  llvm::BumpPtrAllocator alloc;

  unsigned nextVNLAId = 0;
  using VirtualNLAHandles = SmallVector<VirtualNLA *>;

public:
  // Coordinate -> (Virtual) NLA's routing through.
  DenseMap<CoordinateKey, VirtualNLAHandles> pathRoutingTable;
  // TODO: Drop this as unused!! It's helpful for debug printing however :).
  // Module name -> VNLA's rooted.
  DenseMap<StringAttr, VirtualNLAHandles> rootNLAs;
  // NLA sym -> VNLA's
  DenseMap<StringAttr, VirtualNLAHandles> origToVNLAs;
};

} // namespace

void ClassificationPrepass::dump() {
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== ClassificationPrepass Results ===\n";
    // Collect modules in a deterministic order
    SmallVector<Operation *> modules;
    for (auto &op : circuit.getOps()) {
      if (isa<FModuleLike>(op))
        modules.push_back(&op);
    }

    // Print header
    llvm::dbgs() << "Module                          | Inline | Flatten | "
                    "UnderFlatten | Live | HasUnflattenedPath\n";
    llvm::dbgs() << "-------------------------------|--------|---------|----"
                    "----------|------|-------------------\n";

    for (auto *op : modules) {
      auto modName = cast<FModuleLike>(op).getModuleName();
      auto info = modInfoMap.lookup(op);
      llvm::dbgs() << llvm::format("%-30s", modName.str().c_str()) << " |   "
                   << (info.hasInline ? "Y" : " ") << "    |    "
                   << (info.hasFlatten ? "Y" : " ") << "    |      "
                   << (info.underFlatten ? "Y" : " ") << "       |  "
                   << (info.isLive ? "Y" : " ") << "   |    "
                   << (info.hasUnflattenedPath ? "Y" : " ") << "\n";
    }
    llvm::dbgs() << "\n";
  });
}

LogicalResult ClassificationPrepass::run() {
  for (auto &op : circuit.getOps()) {
    // Initialize module information.
    if (auto module = dyn_cast<FModuleLike>(op)) {
      auto &info = modInfoMap[module];
      // TODO: micro-nit: Optimize these scans? 1) single walk 2)reuse
      // StringAttr's for faster comparisons.
      info.hasInline = AnnotationSet::hasAnnotation(module, inlineAnnoClass);
      info.hasFlatten = AnnotationSet::hasAnnotation(module, flattenAnnoClass);
      if (!module.canDiscardOnUseEmpty() ||
          // Modules instantiated by instance-likes that aren't InstanceOp's are
          // considered live.
          llvm::any_of(instanceGraph.lookup(module)->uses(),
                       [&](InstanceRecord *rec) {
                         return !isa<InstanceOp>(rec->getInstance());
                       })) {
        info.markLiveRoot();
      }
      continue;
    }

    // Ignore symbol uses in NLAs.
    if (isa<hw::HierPathOp>(op))
      continue;

    // Mark modules live whose symbols are referenced in other ops.
    auto symbolUses = SymbolTable::getSymbolUses(&op);
    if (!symbolUses)
      continue;
    for (const auto &use : *symbolUses) {
      if (auto flat = dyn_cast<FlatSymbolRefAttr>(use.getSymbolRef()))
        if (auto moduleLike = symbolTable.lookup<FModuleLike>(flat.getAttr()))
          modInfoMap[moduleLike].markLiveRoot();
    }
  }

  instanceGraph.walkInversePostOrder([&](igraph::InstanceGraphNode &node) {
    auto *mod = node.getModule().getOperation();
    auto &modInfo = modInfoMap[mod];

    if (!modInfo.hasUnflattenedPath && !modInfo.underFlatten)
      return;

    bool childrenInstantiated =
        modInfo.hasUnflattenedPath && !modInfo.hasFlatten;

    auto moduleMayBeFlattened = modInfo.underFlatten || modInfo.hasFlatten;
    for (auto *edge : node) {
      auto *childMod = edge->getTarget()->getModule().getOperation();
      auto &childInfo = modInfoMap[childMod];
      bool isRegularModule = isa<FModuleOp>(childMod);
      // TODO: handle earlier, emit diagnostic / return failure.
      assert(
          (isRegularModule || !(childInfo.hasInline || childInfo.hasFlatten)) &&
          "non-fmoduleop with inline/flatten annotation");
      if (isRegularModule && moduleMayBeFlattened)
        childInfo.underFlatten = true;

      if (childrenInstantiated)
        childInfo.hasUnflattenedPath = true;

      if (childInfo.hasUnflattenedPath &&
          (!isRegularModule || !childInfo.hasInline))
        childInfo.isLive = true;
    }
  });

  return success();
}

LogicalResult NLAPrepass::run() {
  for (auto nla : circuit.getBodyBlock()->getOps<hw::HierPathOp>()) {
    auto origSym = nla.getSymNameAttr();
    auto origRoot = nla.root();

    SmallVector<SmallVector<CoordinateKey>> upperPaths;
    SmallVector<CoordinateKey, 8> currentPath;
    if (failed(traceUpUntilSurviving(origRoot, currentPath, upperPaths)))
      return failure();

    SmallVector<CoordinateKey> nlaHops;
    for (auto element : nla.getNamepath()) {
      if (auto ref = dyn_cast<InnerRefAttr>(element))
        nlaHops.push_back({ref.getModule(), ref.getName()});
      else if (auto flat = dyn_cast<FlatSymbolRefAttr>(element))
        nlaHops.push_back({flat.getAttr(), StringAttr()});
      else
        assert(0 && "NLA element must be innerref or flat symbol");
    }

    for (const auto &upperPath : upperPaths) {
      SmallVector<CoordinateKey> absolutePath;
      llvm::append_range(absolutePath, upperPath);
      llvm::append_range(absolutePath, nlaHops);

      processSinglePathContext(origSym, absolutePath);
    }

    // Pre-allocate names and 1-1 churn reduction.
    for (auto &[origSym, realities] : origToVNLAs) {
      if (realities.size() == 1) {
        // Preserve original symbol.
        realities.front()->realizedSym = origSym;
      } else {
        for (auto *vnla : realities) {
          if (!vnla->isLocal()) {
            vnla->realizedSym =
                StringAttr::get(circuit.getContext(),
                                circuitNamespace.newName(origSym.getValue()));
          }
        }
      }
    }
  }

  return success();
}

LogicalResult NLAPrepass::traceUpUntilSurviving(
    StringAttr rootModName, SmallVectorImpl<CoordinateKey> &currentPath,
    SmallVectorImpl<SmallVector<CoordinateKey>> &discoveredPaths) {
  using UseIterator =
      decltype(std::declval<igraph::InstanceGraphNode>().uses().begin());

  struct Frame {
    StringAttr modName;
    UseIterator currentEdge;
    UseIterator endEdge;
    bool isFirstVisit;
  };

  SmallVector<Frame, 16> stack;

  auto pushState = [&](StringAttr name) {
    auto uses = instanceGraph.lookup(name)->uses();
    stack.push_back({name, uses.begin(), uses.end(), /*isFirstVisit=*/true});
  };

  pushState(rootModName);

  // llvm::errs() << "traceUp: " << rootModName << "\n";
  while (!stack.empty()) {
    auto &frame = stack.back();
    // llvm::errs() << "DFS! frame: " << (void*)&frame << ": " << frame.modName
    // << ", isFirst: " << frame.isFirstVisit << "\n";
    if (frame.isFirstVisit) {
      frame.isFirstVisit = false;

      auto *currentModNode = instanceGraph.lookup(frame.modName);
      auto *currentModOp = currentModNode->getModule().getOperation();
      auto info = modInfoMap.lookup(currentModOp);

      // If this module is live in the output in any context, emit VNLA for it.
      if (info.isLive)
        discoveredPaths.push_back(llvm::to_vector(llvm::reverse(currentPath)));

      // If this module is unconditionally live, we're done tracing upwards.
      if (!info.hasInline && !info.underFlatten) {
        stack.pop_back();
        if (!stack.empty())
          currentPath.pop_back();
        continue;
      }
    }
    // If we've exhausted all edges, we're done with this frame.
    if (frame.currentEdge == frame.endEdge) {
      stack.pop_back();
      if (!stack.empty())
        currentPath.pop_back();
      continue;
    }

    // Trace up the current edge and advance it on the frame.
    auto *edge = *frame.currentEdge;
    ++frame.currentEdge;

    auto parentName = edge->getParent()->getModule().getModuleNameAttr();
    currentPath.push_back({parentName, edge->getInstance().getOperation()});
    pushState(parentName);
  }

  return success();
}

void NLAPrepass::processSinglePathContext(
    StringAttr origSym, const SmallVectorImpl<CoordinateKey> &absPath) {
  SmallVector<SurvivingHop> survivingHops;
  assert(!absPath.empty());

  StringAttr activeContainer = absPath.front().first;
  StringAttr currentDest = activeContainer;
  auto *destMod = symbolTable.lookup(currentDest);
  const auto &destInfo = modInfoMap.lookup(destMod);

  bool isTransitiveFlatten = destInfo.hasFlatten;
  for (auto it = absPath.begin(), end = absPath.end(); it != end;) {
    const auto &hop = *it++;
    bool nextHasInline = false;
    bool nextHasFlatten = false;
    StringAttr nextModName;

    bool isTerminal = it == end;
    if (!isTerminal) {
      nextModName = it->first;
      auto *modOp = symbolTable.lookup(nextModName);
      assert(modOp);
      const auto &info = modInfoMap.lookup(modOp);
      nextHasInline = info.hasInline;
      nextHasFlatten = info.hasFlatten;
    }

    bool isEvaporating = !isTerminal && (isTransitiveFlatten || nextHasInline);
    isTransitiveFlatten |= nextHasFlatten;
    if (!isEvaporating) {
      StringAttr sym;
      if (hop.second) { //  && !isTerminal /* drop old-style */) {
        if (auto name = dyn_cast<StringAttr>(hop.second))
          sym = name;
        else if (!isTerminal /* Don't create old-style if not already */) {
          sym = getOrAddInnerSym(
              cast<Operation *>(hop.second),
              [&](FModuleLike mod) -> hw::InnerSymbolNamespace & {
                return modNamespaces.getNamespace(mod);
              });
        }
      }
      StringAttr finalInst;
      if (currentDest == hop.first || isTerminal /* preserve old-style */)
        finalInst = sym;
      survivingHops.push_back({/*origMod=*/hop.first, /*origInst*/ sym,
                               /*finalMod=*/currentDest,
                               /*finalInst=*/finalInst});
      if (nextModName)
        currentDest = nextModName;
    }
  }

  auto *vnla = VirtualNLA::create(alloc, nextVNLAId++, origSym, survivingHops);
  rootNLAs[activeContainer].push_back(vnla);
  origToVNLAs[origSym].push_back(vnla);

  for (const auto &hop : absPath) {
    if (hop.second)
      pathRoutingTable[hop].push_back(vnla);
  }
}

void NLAPrepass::dump() {
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== NLAPrepass Results ===\n";

    // Print rootNLAs section
    llvm::dbgs() << "\nRootNLAs (Module -> VirtualNLAs):\n";

    // Collect and sort module names for deterministic output
    SmallVector<StringAttr> moduleNames;
    for (const auto &[modName, _] : rootNLAs)
      moduleNames.push_back(modName);
    llvm::sort(moduleNames, [](StringAttr a, StringAttr b) {
      return a.getValue() < b.getValue();
    });

    for (auto modName : moduleNames) {
      const auto &vnlas = rootNLAs.lookup(modName);
      llvm::dbgs() << "  Module @" << modName << " (" << vnlas.size()
                   << " VirtualNLAs):\n";
      for (auto *vnla : vnlas)
        vnla->dump();
    }

    // Print pathRoutingTable section
    llvm::dbgs()
        << "\nPath Routing Table (Coordinate -> Active VirtualNLAs):\n";

    // Collect and sort coordinates for deterministic output
    SmallVector<CoordinateKey> coordinates;
    for (const auto &[coord, _] : pathRoutingTable)
      coordinates.push_back(coord);
    llvm::sort(coordinates, [](const CoordinateKey &a, const CoordinateKey &b) {
      // TODO: Simplify into tuple comparison?
      if (a.first != b.first)
        return a.first.getValue() < b.first.getValue();
      // For instance pointer comparison, compare stringified versions
      auto aInst = a.second.dyn_cast<StringAttr>();
      auto bInst = b.second.dyn_cast<StringAttr>();
      if (aInst && bInst)
        return aInst.getValue() < bInst.getValue();
      // If one is Operation* and other is StringAttr, StringAttr comes
      // first
      if (aInst && !bInst)
        return true;
      if (!aInst && bInst)
        return false;
      // Both are Operation*, compare pointers
      return cast<Operation *>(a.second) < cast<Operation *>(b.second);
    });

    for (const auto &coord : coordinates) {
      const auto &vnlas = pathRoutingTable.lookup(coord);
      llvm::dbgs() << "  @" << coord.first;
      if (auto instSym = coord.second.dyn_cast<StringAttr>())
        llvm::dbgs() << "::" << instSym;
      else if (auto *op = coord.second.dyn_cast<Operation *>())
        llvm::dbgs() << "::<op@" << op << ">";
      else
        llvm::dbgs() << "::<none>";

      llvm::dbgs() << " -> [";
      llvm::interleaveComma(vnlas, llvm::dbgs(), [&](VirtualNLA *vnla) {
        llvm::dbgs() << "#" << vnla->id;
      });
      llvm::dbgs() << "]\n";
    }

    llvm::dbgs() << "\n";
  });
}

//===----------------------------------------------------------------------===//
// Module Inlining Support
//===----------------------------------------------------------------------===//

/// This function is used after inlining a module, to handle the conversion
/// between module ports and instance results. This maps each wire to the
/// result of the instance operation.  When future operations are cloned from
/// the current block, they will use the value of the wire instead of the
/// instance results.
static void mapResultsToWires(IRMapping &mapper, SmallVectorImpl<Value> &wires,
                              InstanceOp instance) {
  for (auto [result, wire] : llvm::zip_equal(instance.getResults(), wires))
    mapper.map(result, wire);
}

/// Process each operation, updating InnerRefAttr's using the specified map
/// and the given name as the containing IST of the mapped-to sym names.
static void replaceInnerRefUsers(ArrayRef<Operation *> newOps,
                                 const InnerRefToNewNameMap &map,
                                 StringAttr istName) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](hw::InnerRefAttr innerRef) {
    auto it = map.find(innerRef);
    // TODO: what to do with users that aren't local (or not mapped?).
    assert(it != map.end());

    return std::pair{hw::InnerRefAttr::get(istName, it->second),
                     WalkResult::skip()};
  });
  llvm::for_each(newOps,
                 [&](auto *op) { replacer.recursivelyReplaceElementsIn(op); });
}

/// Generate and creating map entries for new inner symbol based on old one
/// and an appropriate namespace for creating unique names for each.
static hw::InnerSymAttr uniqueInNamespace(hw::InnerSymAttr old,
                                          InnerRefToNewNameMap &map,
                                          hw::InnerSymbolNamespace &ns,
                                          StringAttr istName) {
  if (!old || old.empty())
    return old;

  bool anyChanged = false;

  SmallVector<hw::InnerSymPropertiesAttr> newProps;
  auto *context = old.getContext();
  for (auto &prop : old) {
    auto newSym = ns.newName(prop.getName().strref());
    if (newSym == prop.getName()) {
      newProps.push_back(prop);
      continue;
    }
    auto newSymStrAttr = StringAttr::get(context, newSym);
    auto newProp = hw::InnerSymPropertiesAttr::get(
        context, newSymStrAttr, prop.getFieldID(), prop.getSymVisibility());
    anyChanged = true;
    newProps.push_back(newProp);
  }

  auto newSymAttr = anyChanged ? hw::InnerSymAttr::get(context, newProps) : old;

  for (auto [oldProp, newProp] : llvm::zip(old, newSymAttr)) {
    assert(oldProp.getFieldID() == newProp.getFieldID());
    // Map InnerRef to this inner sym -> new inner sym.
    map[hw::InnerRefAttr::get(istName, oldProp.getName())] = newProp.getName();
  }

  return newSymAttr;
}

//===----------------------------------------------------------------------===//
// Inliner
//===----------------------------------------------------------------------===//

/// Inlines, flattens, and removes dead modules in a circuit.
///
/// The inliner works in a top down fashion, in parents-before-children order,
/// visiting only live modules and inlines every possible instance.
/// With this method of recursive top-down inlining, each operation will be
/// cloned directly to its final location.
///
/// Modules are initially marked live if they are public or referenced by
/// non-module operations. Additional modules become live as they are discovered
/// during inlining (when a live module instantiates them after
/// inlining/flattening). Dead modules are removed at the end.
///
/// During the inlining process, every cloned operation with a name must be
/// prefixed with the instance's name. The top-down process means that we know
/// the entire desired prefix when we clone an operation, and can set the name
/// attribute once. This means that we will not create any intermediate name
/// attributes (which will be interned by the compiler), and helps keep down the
/// total memory usage.
namespace {
class Inliner {
public:
  /// Initialize the inliner to run on this circuit.
  Inliner(CircuitOp circuit, SymbolTable &symbolTable,
          InstanceGraph &instanceGraph, CircuitNamespace &circuitNamespace,
          ModNamespaces &modNamespaces,
          ClassificationPrepass &classificationPrepass, NLAPrepass &nlaPrepass);

  /// Run the inliner.
  LogicalResult run();

private:
  /// Inlining context, one per module being inlined into.
  /// Cleans up backedges on destruction.
  struct ModuleInliningContext {
    ModuleInliningContext(FModuleOp module,
                          hw::InnerSymbolNamespace &modNamespace)
        : module(module), modNamespace(modNamespace), b(module.getContext()) {}
    /// Top-level module for current inlining task.
    FModuleOp module;
    /// Namespace for generating new names in `module`.
    hw::InnerSymbolNamespace &modNamespace;
    /// Builder, insertion point into module.
    OpBuilder b;
  };

  /// One inlining level, created for each instance inlined or flattened.
  /// All inner symbols renamed are recorded in relocatedInnerSyms,
  /// and new operations in newOps.  On destruction newOps are fixed up.
  struct InliningLevel {
    InliningLevel(ModuleInliningContext &mic, FModuleOp childModule)
        : mic(mic), childModule(childModule) {}

    /// Top-level inlining context.
    ModuleInliningContext &mic;
    /// Map of inner-refs to the new inner sym.
    InnerRefToNewNameMap relocatedInnerSyms;
    /// All operations cloned are tracked here.
    SmallVector<Operation *> newOps;
    /// Wires and other values introduced for ports.
    SmallVector<Value> wires;
    /// The module being inlined (this "level").
    FModuleOp childModule;
    /// The explicit debug scope of the inlined instance.
    Value debugScope;
    /// VNLA's active at this level
    SmallVector<VirtualNLA *> activeNLAs;
    // TODO: Sort activeNLA's, DenseMap can point to ranges within!
    DenseMap<StringAttr, SmallVector<VirtualNLA *>> activeNLAsBySym;

    void setActivePaths(ArrayRef<VirtualNLA *> nlas) {
      activeNLAs.assign(nlas);
      for (auto *nla : activeNLAs)
        activeNLAsBySym[nla->origSym].push_back(nla);
    }

    ~InliningLevel() {
      replaceInnerRefUsers(newOps, relocatedInnerSyms,
                           mic.module.getNameAttr());
    }
  };

  /// Returns true if the NLA matches the current path.  This will only return
  /// false if there is a mismatch indicating that the NLA definitely is
  /// referring to some other path.
  // bool doesNLAMatchCurrentPath(hw::HierPathOp nla);

  /// Rename an operation and unique any symbols it has.
  /// Returns true iff symbol was changed.
  bool rename(StringRef prefix, Operation *op, InliningLevel &il);

  /// Rename an InstanceOp and unique any symbols it has.
  /// Requires old and new operations to appropriately update the `HierPathOp`'s
  /// that it participates in.
  bool renameInstance(StringRef prefix, InliningLevel &il, InstanceOp oldInst,
                      InstanceOp newInst);

  /// Clone and rename an operation.  Insert the operation into the inlining
  /// level.
  void cloneAndRename(StringRef prefix, InliningLevel &il, IRMapping &mapper,
                      Operation &op);

  /// TODO: Comment
  SmallVector<Attribute> computeNewAnnotations(AnnotationSet annos,
                                               const InliningLevel &il);

  /// TODO: Comment
  // Update VirtualNLA leaf symbols if the port symbol was renamed.
  // (needed for old-style NLAs, remove this as soon as we can drop support!)
  void updateVirtualNLALeafSymbols(Inliner::InliningLevel &il,
                                   hw::InnerSymAttr oldSymAttr,
                                   hw::InnerSymAttr newSymAttr);

  /// TODO: Comment
  void setActiveNLAsForChild(std::optional<ArrayRef<VirtualNLA *>> activeNLAs,
                             StringAttr moduleName, InliningLevel &childIL,
                             Operation *instance);

  /// Rewrite the ports of a module as wires.  This is similar to
  /// cloneAndRename, but operating on ports.
  /// Wires are added to il.wires.
  void mapPortsToWires(StringRef prefix, InliningLevel &il, IRMapping &mapper);

  /// Returns true if the operation is annotated to be flattened.
  bool shouldFlatten(Operation *op);

  /// Returns true if the operation is annotated to be inlined.
  bool shouldInline(Operation *op);

  /// Check not inlining into anything other than layerblock or module.
  /// In the future, could check this per-inlined-operation.
  LogicalResult checkInstanceParents(InstanceOp instance);

  /// Walk the specified block, invoking `process` for operations visited
  /// forward+pre-order.  Handles cloning supported operations with regions,
  /// so that `process` is only invoked on regionless operations.
  LogicalResult
  inliningWalk(OpBuilder &builder, Block *block, IRMapping &mapper,
               llvm::function_ref<LogicalResult(Operation *op)> process);

  /// Flattens a target module into the insertion point of the builder,
  /// renaming all operations using the prefix.  This clones all operations from
  /// the target, and does not trigger inlining on the target itself.
  LogicalResult flattenInto(StringRef prefix, InliningLevel &il,
                            IRMapping &mapper);

  /// Inlines a target module into the insertion point of the builder,
  /// prefixing all operations with prefix.  This clones all operations from
  /// the target, and does not trigger inlining on the target itself.
  LogicalResult inlineInto(StringRef prefix, InliningLevel &il,
                           IRMapping &mapper);

  /// Recursively flatten all instances in a module.
  LogicalResult flattenInstances(FModuleOp module);

  /// Inline any instances in the module which were marked for inlining.
  LogicalResult inlineInstances(FModuleOp module);

  /// Create a debug scope for an inlined instance at the current insertion
  /// point of the `il.mic` builder.
  void createDebugScope(InliningLevel &il, InstanceOp instance,
                        Value parentScope = {});

  /// Identify all module-only NLA's, marking their MutableNLA's accordingly.
  // void identifyNLAsTargetingOnlyModules();

  CircuitOp circuit;
  MLIRContext *context;

  // A symbol table with references to each module in a circuit.
  SymbolTable &symbolTable;

  // The original instance graph.  This is NOT maintained if mutations are made.
  InstanceGraph &instanceGraph;

  // Namespaces
  CircuitNamespace &circuitNamespace;
  ModNamespaces &modNamespaces;

  // Prepass results
  ClassificationPrepass &classificationPrepass;
  NLAPrepass &nlaPrepass;

  /// The debug scopes created for inlined instances. Scopes that are unused
  /// after inlining will be deleted again.
  SmallVector<debug::ScopeOp> debugScopes;
};
} // namespace

/// If this operation or any child operation has a name, add the prefix to that
/// operation's name.  If the operation has any inner symbols, make sure that
/// these are unique in the namespace.  Record renamed inner symbols
/// in relocatedInnerSyms map for renaming local users.
bool Inliner::rename(StringRef prefix, Operation *op, InliningLevel &il) {
  // Debug operations with implicit module scope now need an explicit scope,
  // since inlining has destroyed the module whose scope they implicitly used.
  auto updateDebugScope = [&](auto op) {
    if (!op.getScope())
      op.getScopeMutable().assign(il.debugScope);
  };
  if (auto varOp = dyn_cast<debug::VariableOp>(op))
    return updateDebugScope(varOp), false;
  if (auto scopeOp = dyn_cast<debug::ScopeOp>(op))
    return updateDebugScope(scopeOp), false;

  // Add a prefix to things that has a "name" attribute.
  if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
    op->setAttr("name", StringAttr::get(op->getContext(),
                                        (prefix + nameAttr.getValue())));

  // If the operation has an inner symbol, ensure that it is unique.  Record
  // renames for any NLAs that this participates in if the symbol was renamed.
  auto symOp = dyn_cast<hw::InnerSymbolOpInterface>(op);
  if (!symOp)
    return false;
  auto oldSymAttr = symOp.getInnerSymAttr();
  auto newSymAttr =
      uniqueInNamespace(oldSymAttr, il.relocatedInnerSyms, il.mic.modNamespace,
                        il.childModule.getNameAttr());

  if (!newSymAttr)
    return false;

  // TODO: Gate this on whether this partcipates in NLA's to avoid
  // unnecessary scanning.
  updateVirtualNLALeafSymbols(il, oldSymAttr, newSymAttr);
  symOp.setInnerSymbolAttr(newSymAttr);

  return newSymAttr != oldSymAttr;
}

bool Inliner::renameInstance(StringRef prefix, InliningLevel &il,
                             InstanceOp oldInst, InstanceOp newInst) {
  // TODO: There is currently no good way to annotate an explicit parent scope
  // on instances. Just emit a note in debug runs until this is resolved.
  LLVM_DEBUG({
    if (il.debugScope)
      llvm::dbgs() << "Discarding parent debug scope for " << oldInst << "\n";
  });

  auto oldInstSym = getInnerSymName(oldInst);
  auto symbolChanged = rename(prefix, newInst, il);
  auto newSymAttr = getInnerSymName(newInst);

  if (oldInstSym /*&& symbolChanged*/) {
    assert(newSymAttr);
    StringAttr origMod = il.childModule.getModuleNameAttr();
    StringAttr destMod = il.mic.module.getModuleNameAttr();
    for (auto *nla : il.activeNLAs) {
      for (auto &hop : nla->getPathMutable()) {
        if (hop.origMod == origMod && hop.origInst == oldInstSym &&
            hop.finalMod == destMod) {
          hop.finalInst = newSymAttr;
        }
      }
    }
  }
  return symbolChanged;
}

SmallVector<Attribute> Inliner::computeNewAnnotations(AnnotationSet annos,
                                                      const InliningLevel &il) {
  SmallVector<Attribute> newAnnotations;
  for (auto anno : annos) {
    auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
    // If not an NLA, keep it.
    if (!sym) {
      newAnnotations.push_back(anno.getAttr());
      continue;
    }

    auto it = il.activeNLAsBySym.find(sym.getAttr());

    // If not active, remove it!
    if (it == il.activeNLAsBySym.end())
      continue;

    // Otherwise preserve it, cloning / modifying per active VNLAs.
    for (auto *matched : it->second) {
      if (matched->isLocal()) {
        Annotation copy(anno);
        copy.removeMember("circt.nonlocal");
        newAnnotations.push_back(copy.getAttr());
        continue;
      }
      // TODO: Unconditionally set in this loop? Local is "used".
      matched->wasUsed = true;

      // Keep as-is if we're using the same symbol as original.
      if (matched->realizedSym == matched->origSym) {
        assert(llvm::hasSingleElement(it->second));
        newAnnotations.push_back(anno.getAttr());
        continue;
      }
      // Otherwise modify to use new symbol.
      Annotation copy(anno);
      copy.setMember("circt.nonlocal",
                     FlatSymbolRefAttr::get(matched->realizedSym));
      newAnnotations.push_back(copy.getAttr());
    }
  }
  return newAnnotations;
}

void Inliner::updateVirtualNLALeafSymbols(Inliner::InliningLevel &il,
                                          hw::InnerSymAttr oldSymAttr,
                                          hw::InnerSymAttr newSymAttr) {
  // TODO: We can skip scanning if we grab the NLA symbols this target
  // participates in during our computeNewAnnotations work, and then directly
  // jump to the VNLA's that might have leaf references for this using
  // il.activeNLAsBySym!
  if (oldSymAttr && oldSymAttr != newSymAttr) {
    assert(newSymAttr);
    StringAttr origMod = il.childModule.getModuleNameAttr();
    StringAttr destMod = il.mic.module.getModuleNameAttr();
    for (auto *nla : il.activeNLAs) {
      if (nla->isLocal())
        continue;
      auto &last = nla->getPathMutable().back();
      if (last.origMod == origMod && last.finalMod == destMod) {
        for (auto prop : oldSymAttr.getProps()) {
          if (last.origInst == prop.getName()) {
            last.finalInst = newSymAttr.getSymIfExists(prop.getFieldID());
            break;
          }
        }
      }
    }
  }
}

void Inliner::setActiveNLAsForChild(
    std::optional<ArrayRef<VirtualNLA *>> activeNLAs, StringAttr moduleName,
    InliningLevel &childIL, Operation *instance) {
  if (activeNLAs && activeNLAs->empty())
    return;
  SmallVector<VirtualNLA *> instNLAs;

  auto instInnerSym = getInnerSymName(instance);
  // Lookup using inner symbol as well as instance op itself.
  // TODO: Do better! :)
  // TODO: Less copying, ensure efficient-ish intersection/filtering
  if (auto it = nlaPrepass.pathRoutingTable.find({moduleName, instInnerSym});
      it != nlaPrepass.pathRoutingTable.end())
    llvm::append_range(instNLAs, it->second);
  if (auto it = nlaPrepass.pathRoutingTable.find({moduleName, instance});
      it != nlaPrepass.pathRoutingTable.end())
    llvm::append_range(instNLAs, it->second);

  // TODO: better
  auto sortVNLAs = [](const VirtualNLA *a, const VirtualNLA *b) {
    return a->id < b->id;
  };
  llvm::sort(instNLAs, sortVNLAs);
  instNLAs.erase(llvm::unique(instNLAs), instNLAs.end());
  if (activeNLAs) {
    SmallVector<VirtualNLA *> childActiveNLAs;
    std::set_intersection(instNLAs.begin(), instNLAs.end(), activeNLAs->begin(),
                          activeNLAs->end(),
                          std::back_inserter(childActiveNLAs), sortVNLAs);
    childIL.setActivePaths(childActiveNLAs);
  } else {
    childIL.setActivePaths(instNLAs);
  }
}

/// This function is used before inlining a module, to handle the conversion
/// between module ports and instance results. For every port in the target
/// module, create a wire, and assign a mapping from each module port to the
/// wire. When the body of the module is cloned, the value of the wire will be
/// used instead of the module's ports.
void Inliner::mapPortsToWires(StringRef prefix, InliningLevel &il,
                              IRMapping &mapper) {
  auto target = il.childModule;
  auto portInfo = target.getPorts();
  for (unsigned i = 0, e = target.getNumPorts(); i < e; ++i) {
    auto arg = target.getArgument(i);
    // Get the type of the wire.
    auto type = type_cast<FIRRTLType>(arg.getType());

    // Compute new symbols if needed.
    auto oldSymAttr = portInfo[i].sym;
    auto newSymAttr =
        uniqueInNamespace(oldSymAttr, il.relocatedInnerSyms,
                          il.mic.modNamespace, target.getNameAttr());

    SmallVector<Attribute> newAnnotations =
        computeNewAnnotations(AnnotationSet::forPort(target, i), il);
    if (!newAnnotations.empty()) // Not quite right but sufficient.
      updateVirtualNLALeafSymbols(il, oldSymAttr, newSymAttr);

    Value wire =
        WireOp::create(
            il.mic.b, target.getLoc(), type,
            StringAttr::get(context, (prefix + portInfo[i].getName())),
            NameKindEnumAttr::get(context, NameKindEnum::DroppableName),
            ArrayAttr::get(context, newAnnotations), newSymAttr,
            /*forceable=*/UnitAttr{})
            .getResult();
    il.wires.push_back(wire);
    mapper.map(arg, wire);
  }
}

/// Clone an operation, mapping used values and results with the mapper, and
/// apply the prefix to the name of the operation. This will clone to the
/// insert point of the builder.  Insert the operation into the level.
void Inliner::cloneAndRename(StringRef prefix, InliningLevel &il,
                             IRMapping &mapper, Operation &op) {
  // Strip any non-local annotations which are local.
  AnnotationSet oldAnnotations(&op);
  auto newAnnotations = computeNewAnnotations(oldAnnotations, il);

  // Clone and rename.
  assert(op.getNumRegions() == 0 &&
         "operation with regions should not reach cloneAndRename");
  auto *newOp = il.mic.b.cloneWithoutRegions(op, mapper);

  // Rename the new operation.
  // (add prefix to it, if named, and unique-ify symbol, updating NLA's).

  // Instances require extra handling to update HierPathOp's if their symbols
  // change.
  if (auto oldInst = dyn_cast<InstanceOp>(op))
    renameInstance(prefix, il, oldInst, cast<InstanceOp>(newOp));
  else
    rename(prefix, newOp, il);

  // We want to avoid attaching an empty annotation array on to an op that
  // never had an annotation array in the first place.
  if (!newAnnotations.empty() || !oldAnnotations.empty())
    AnnotationSet(newAnnotations, context).applyToOperation(newOp);

  il.newOps.push_back(newOp);
}

bool Inliner::shouldFlatten(Operation *op) {
  return classificationPrepass.getModInfoMap().lookup(op).hasFlatten;
}

bool Inliner::shouldInline(Operation *op) {
  return classificationPrepass.getModInfoMap().lookup(op).hasInline;
}

LogicalResult Inliner::inliningWalk(
    OpBuilder &builder, Block *block, IRMapping &mapper,
    llvm::function_ref<LogicalResult(Operation *op)> process) {
  struct IPs {
    OpBuilder::InsertPoint target;
    Block::iterator source;
  };
  // Invariant: no Block::iterator == end(), can't getBlock().
  SmallVector<IPs> inliningStack;
  if (block->empty())
    return success();

  inliningStack.push_back(IPs{builder.saveInsertionPoint(), block->begin()});
  OpBuilder::InsertionGuard guard(builder);

  while (!inliningStack.empty()) {
    auto target = inliningStack.back().target;
    builder.restoreInsertionPoint(target);
    Operation *source;
    // Get next source operation.
    {
      auto &ips = inliningStack.back();
      source = &*ips.source;
      auto end = source->getBlock()->end();
      if (++ips.source == end)
        inliningStack.pop_back();
    }

    // Does the source have regions? If not, use callback to process.
    if (source->getNumRegions() == 0) {
      assert(builder.saveInsertionPoint().getPoint() == target.getPoint());
      // Clone source into insertion point 'target'.
      if (failed(process(source)))
        return failure();
      assert(builder.saveInsertionPoint().getPoint() == target.getPoint());

      continue;
    }

    // Limited support for region-containing operations.
    if (!isa<LayerBlockOp, WhenOp, MatchOp>(source))
      return source->emitError("unsupported operation '")
             << source->getName() << "' cannot be inlined";

    // Note: This does not use cloneAndRename for simplicity,
    // as there are no annotations, symbols to rename, or names
    // to prefix.  This does mean these operations do not appear
    // in `il.newOps` for inner-ref renaming walk, FWIW.
    auto *newOp = builder.cloneWithoutRegions(*source, mapper);
    for (auto [newRegion, oldRegion] : llvm::reverse(
             llvm::zip_equal(newOp->getRegions(), source->getRegions()))) {
      // If region has no blocks, skip.
      if (oldRegion.empty()) {
        assert(newRegion.empty());
        continue;
      }
      // Otherwise, assert single block.  Multiple blocks is trickier.
      assert(oldRegion.hasOneBlock());

      // Create new block and add to inlining stack for processing.
      auto &oldBlock = oldRegion.getBlocks().front();
      auto &newBlock = newRegion.emplaceBlock();
      mapper.map(&oldBlock, &newBlock);

      // Copy block arguments, and add mapping for each.
      for (auto arg : oldBlock.getArguments())
        mapper.map(arg, newBlock.addArgument(arg.getType(), arg.getLoc()));

      if (oldBlock.empty())
        continue;

      inliningStack.push_back(
          IPs{OpBuilder::InsertPoint(&newBlock, newBlock.begin()),
              oldBlock.begin()});
    }
  }
  return success();
}

LogicalResult Inliner::checkInstanceParents(InstanceOp instance) {
  auto *parent = instance->getParentOp();
  while (!isa<FModuleLike>(parent)) {
    if (!isa<LayerBlockOp>(parent))
      return instance->emitError("cannot inline instance")
                 .attachNote(parent->getLoc())
             << "containing operation '" << parent->getName()
             << "' not safe to inline into";
    parent = parent->getParentOp();
  }
  return success();
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult Inliner::flattenInto(StringRef prefix, InliningLevel &il,
                                   IRMapping &mapper) {
  auto target = il.childModule;
  auto moduleName = target.getNameAttr();

  LLVM_DEBUG(llvm::dbgs() << "flattening " << target.getModuleName() << " into "
                          << il.mic.module.getModuleName() << "\n");
  auto visit = [&](Operation *op) {
    // If it's not an instance op, clone it and continue.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance) {
      // if (auto instanceLike = dyn_cast<FInstanceLike>(op))
      //   markUnknownFInstanceLikeModulesLive(instanceLike);
      cloneAndRename(prefix, il, mapper, *op);
      return success();
    }

    // If it's not a regular module we can't inline it. Mark it as live.
    auto *moduleOp = symbolTable.lookup(instance.getModuleName());
    auto childModule = dyn_cast<FModuleOp>(moduleOp);
    if (!childModule) {
      assert(classificationPrepass.getModInfoMap().lookup(moduleOp).isLive);
      cloneAndRename(prefix, il, mapper, *op);
      return success();
    }

    if (failed(checkInstanceParents(instance)))
      return failure();

    InliningLevel childIL(il.mic, childModule);
    setActiveNLAsForChild(il.activeNLAs, moduleName, childIL, instance);
    createDebugScope(childIL, instance, il.debugScope);

    // Create the wire mapping for results + ports.
    auto nestedPrefix = (prefix + instance.getName() + "_").str();
    mapPortsToWires(nestedPrefix, childIL, mapper);
    mapResultsToWires(mapper, childIL.wires, instance);

    // Unconditionally flatten all instance operations.
    if (failed(flattenInto(nestedPrefix, childIL, mapper)))
      return failure();
    return success();
  };
  return inliningWalk(il.mic.b, target.getBodyBlock(), mapper, visit);
}

LogicalResult Inliner::flattenInstances(FModuleOp module) {
  auto moduleName = module.getNameAttr();
  ModuleInliningContext mic(module, modNamespaces.getNamespace(module));

  LLVM_DEBUG(llvm::dbgs() << "inlining instances within " << moduleName
                          << "...\n");
  auto visit = [&](FInstanceLike instanceLike) {
    auto instance = dyn_cast<InstanceOp>(*instanceLike);
    if (!instance) {
      // markUnknownFInstanceLikeModulesLive(instanceLike);
      return WalkResult::advance();
    }
    // If it's not a regular module we can't inline it. Mark it as live.
    auto *targetModule = symbolTable.lookup(instance.getModuleName());
    auto target = dyn_cast<FModuleOp>(targetModule);
    if (!target) {
      assert(classificationPrepass.getModInfoMap().lookup(targetModule).isLive);
      return WalkResult::advance();
    }

    if (failed(checkInstanceParents(instance)))
      return WalkResult::interrupt();

    // Create the wire mapping for results + ports. We RAUW the results instead
    // of mapping them.
    IRMapping mapper;
    mic.b.setInsertionPoint(instance);

    InliningLevel il(mic, target);
    setActiveNLAsForChild(/* Activate all through this instance */ std::nullopt,
                          moduleName, il, instance);
    createDebugScope(il, instance);

    auto nestedPrefix = (instance.getName() + "_").str();
    mapPortsToWires(nestedPrefix, il, mapper);
    for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i)
      instance.getResult(i).replaceAllUsesWith(il.wires[i]);

    // Recursively flatten the target module.
    if (failed(flattenInto(nestedPrefix, il, mapper)))
      return WalkResult::interrupt();

    // Erase the replaced instance.
    instance.erase();
    return WalkResult::skip();
  };
  return failure(module.getBodyBlock()
                     ->walk<mlir::WalkOrder::PreOrder>(visit)
                     .wasInterrupted());
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult Inliner::inlineInto(StringRef prefix, InliningLevel &il,
                                  IRMapping &mapper) {
  auto target = il.childModule;
  auto moduleName = target.getNameAttr();

  LLVM_DEBUG(llvm::dbgs() << "inlining " << target.getModuleName() << " into "
                          << il.mic.module.getModuleName() << "\n");

  auto visit = [&](Operation *op) {
    // If it's not an instance op, clone it and continue.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance) {
      cloneAndRename(prefix, il, mapper, *op);
      return success();
    }

    // If it's not a regular module we can't inline it. Mark it as live.
    auto *moduleOp = symbolTable.lookup(instance.getModuleName());
    auto childModule = dyn_cast<FModuleOp>(moduleOp);
    if (!childModule) {
      assert(classificationPrepass.getModInfoMap().lookup(moduleOp).isLive);
      cloneAndRename(prefix, il, mapper, *op);
      return success();
    }

    // If we aren't inlining the target, mark it live.
    if (!shouldInline(childModule)) {
      assert(classificationPrepass.getModInfoMap().lookup(childModule).isLive);
      cloneAndRename(prefix, il, mapper, *op);
      return success();
    }

    if (failed(checkInstanceParents(instance)))
      return failure();

    InliningLevel childIL(il.mic, childModule);
    setActiveNLAsForChild(il.activeNLAs, moduleName, childIL, instance);
    createDebugScope(childIL, instance, il.debugScope);

    // Create the wire mapping for results + ports.
    auto nestedPrefix = (prefix + instance.getName() + "_").str();
    mapPortsToWires(nestedPrefix, childIL, mapper);
    mapResultsToWires(mapper, childIL.wires, instance);

    if (shouldFlatten(childModule)) {
      if (failed(flattenInto(nestedPrefix, childIL, mapper)))
        return failure();
    } else {
      if (failed(inlineInto(nestedPrefix, childIL, mapper)))
        return failure();
    }
    return success();
  };

  return inliningWalk(il.mic.b, target.getBodyBlock(), mapper, visit);
}

LogicalResult Inliner::inlineInstances(FModuleOp module) {
  auto moduleName = module.getNameAttr();
  ModuleInliningContext mic(module, modNamespaces.getNamespace(module));

  LLVM_DEBUG(llvm::dbgs() << "inlining instances within " << moduleName
                          << "...\n");
  auto visit = [&](FInstanceLike instanceLike) {
    auto instance = dyn_cast<InstanceOp>(*instanceLike);
    if (!instance) {
      // markUnknownFInstanceLikeModulesLive(instanceLike);
      return WalkResult::advance();
    }
    // If it's not a regular module we can't inline it. Mark it as live.
    auto *childModule = symbolTable.lookup(instance.getModuleName());
    auto target = dyn_cast<FModuleOp>(childModule);
    if (!target) {
      assert(classificationPrepass.getModInfoMap().lookup(childModule).isLive);
      return WalkResult::advance();
    }

    // If we aren't inlining the target, mark it live.
    if (!shouldInline(target)) {
      return WalkResult::advance();
    }

    if (failed(checkInstanceParents(instance)))
      return WalkResult::interrupt();

    // TODO: cache modInfoMap lookup and reuse ModInfo info?

    // Create the wire mapping for results + ports. We RAUW the results instead
    // of mapping them.
    IRMapping mapper;
    mic.b.setInsertionPoint(instance);
    auto nestedPrefix = (instance.getName() + "_").str();

    InliningLevel childIL(mic, target);
    setActiveNLAsForChild(/* Activate all through this instance */ std::nullopt,
                          moduleName, childIL, instance);
    createDebugScope(childIL, instance);

    mapPortsToWires(nestedPrefix, childIL, mapper);
    for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i)
      instance.getResult(i).replaceAllUsesWith(childIL.wires[i]);

    // Inline the module, it can be marked as flatten and inline.
    if (shouldFlatten(target)) {
      if (failed(flattenInto(nestedPrefix, childIL, mapper)))
        return WalkResult::interrupt();
    } else {
      // Recursively inline all the child modules under `parent`, that are
      // marked to be inlined.
      if (failed(inlineInto(nestedPrefix, childIL, mapper)))
        return WalkResult::interrupt();
    }

    // Erase the replaced instance.
    instance.erase();
    return WalkResult::skip();
  };

  return failure(module.getBodyBlock()
                     ->walk<mlir::WalkOrder::PreOrder>(visit)
                     .wasInterrupted());
}

void Inliner::createDebugScope(InliningLevel &il, InstanceOp instance,
                               Value parentScope) {
  auto op = debug::ScopeOp::create(
      il.mic.b, instance.getLoc(), instance.getInstanceNameAttr(),
      instance.getModuleNameAttr().getAttr(), parentScope);
  debugScopes.push_back(op);
  il.debugScope = op;
}

#if 0
void Inliner::identifyNLAsTargetingOnlyModules() {
  DenseSet<Operation *> nlaTargetedModules;

  // Identify candidate NLA's: those that end in a module
  for (auto &[sym, mnla] : nlaMap) {
    auto nla = mnla.getNLA();
    if (nla.isModule()) {
      auto mod = symbolTable.lookup<FModuleLike>(nla.leafMod());
      assert(mod &&
             "NLA ends in module reference but does not target FModuleLike?");
      nlaTargetedModules.insert(mod);
    }
  }

  // Helper to scan leaf modules for users of NLAs, gathering by symbol names
  auto scanForNLARefs = [&](FModuleLike mod) {
    DenseSet<StringAttr> referencedNLASyms;
    auto scanAnnos = [&](const AnnotationSet &annos) {
      for (auto anno : annos)
        if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal"))
          referencedNLASyms.insert(sym.getAttr());
    };
    // Scan ports
    for (unsigned i = 0, e = mod.getNumPorts(); i != e; ++i)
      scanAnnos(AnnotationSet::forPort(mod, i));

    // Scan operations (and not the module itself):
    // (Walk includes module for lack of simple/generic way to walk body only)
    mod.walk([&](Operation *op) {
      if (op == mod.getOperation())
        return;
      scanAnnos(AnnotationSet(op));

      // Check MemOp and InstanceOp port annotations, special case
      TypeSwitch<Operation *>(op).Case<MemOp, InstanceOp>([&](auto op) {
        for (auto portAnnoAttr : op.getPortAnnotations())
          scanAnnos(AnnotationSet(cast<ArrayAttr>(portAnnoAttr)));
      });
    });

    return referencedNLASyms;
  };

  // Reduction operator
  auto mergeSets = [](auto &&a, auto &&b) {
    a.insert(b.begin(), b.end());
    return std::move(a);
  };

  // Walk modules in parallel, scanning for references to NLA's
  // Gather set of NLA's referenced by each module's ports/operations.
  SmallVector<FModuleLike, 0> mods(nlaTargetedModules.begin(),
                                   nlaTargetedModules.end());
  auto nonModOnlyNLAs =
      transformReduce(circuit->getContext(), mods, DenseSet<StringAttr>{},
                      mergeSets, scanForNLARefs);

  // Mark NLA's that were not referenced as module-only
  for (auto &[_, mnla] : nlaMap) {
    auto nla = mnla.getNLA();
    if (nla.isModule() && !nonModOnlyNLAs.count(nla.getSymNameAttr()))
      mnla.markModuleOnly();
  }
}
#endif

Inliner::Inliner(CircuitOp circuit, SymbolTable &symbolTable,
                 InstanceGraph &instanceGraph,
                 CircuitNamespace &circuitNamespace,
                 ModNamespaces &modNamespaces,
                 ClassificationPrepass &classificationPrepass,
                 NLAPrepass &nlaPrepass)
    : circuit(circuit), context(circuit.getContext()), symbolTable(symbolTable),
      instanceGraph(instanceGraph), circuitNamespace(circuitNamespace),
      modNamespaces(modNamespaces),
      classificationPrepass(classificationPrepass), nlaPrepass(nlaPrepass) {}

LogicalResult Inliner::run() {
  // TODO: Circle back to check module-only NLAs.

  // Populate module list with ALL modules in parents-before-children order.
  // We will only visit those known to be live, partially dynamically
  // discovered. This ensures overall we still process parents strictly before
  // children.
  SmallVector<FModuleOp, 16> modules;
  instanceGraph.walkInversePostOrder([&](igraph::InstanceGraphNode &node) {
    if (auto module = dyn_cast<FModuleOp>(*node.getModule()))
      modules.push_back(module);
  });

  // If the module is marked for flattening, flatten it. Otherwise, inline
  // every instance marked to be inlined.
  for (auto moduleOp : modules) {
    // Skip modules we haven't determined to be 'live' so far.
    if (!classificationPrepass.getModInfoMap().lookup(moduleOp).isLive)
      continue;
    if (shouldFlatten(moduleOp)) {
      if (failed(flattenInstances(moduleOp)))
        return failure();
      // TODO: This seems legacy due to not IPO, we should just rip
      // flatten/inline off as we go! Delete the flatten annotation, the
      // transform was performed. Even if visited again in our walk (for
      // inlining), we've just flattened it and so the annotation is no longer
      // needed.
      AnnotationSet::removeAnnotations(moduleOp, flattenAnnoClass);
    } else {
      if (failed(inlineInstances(moduleOp)))
        return failure();
    }
  }

  // Delete debug scopes that ended up being unused. Erase them in reverse order
  // since scopes at the back may have uses on scopes at the front.
  for (auto scopeOp : llvm::reverse(debugScopes))
    if (scopeOp.use_empty())
      scopeOp.erase();
  debugScopes.clear();

  // Delete all unreferenced (post-inlining) modules.
  for (auto mod : llvm::make_early_inc_range(
           circuit.getBodyBlock()->getOps<FModuleLike>())) {
    if (classificationPrepass.getModInfoMap().lookup(mod).isLive)
      continue;
    mod.erase();
  }

  // Remove leftover inline annotations, and check no flatten annotations
  // remain as they should have been processed and removed.
  for (auto mod : circuit.getBodyBlock()->getOps<FModuleLike>()) {
    if (shouldInline(mod)) {
      assert(mod.isPublic() &&
             "non-public module with inline annotation still present");
      AnnotationSet::removeAnnotations(mod, inlineAnnoClass);
    }
  }

  // Sweep and writeback!
  auto rewriteAnnos = [&](AnnotationSet &annos, StringAttr modName,
                          SmallVectorImpl<Attribute> &newAnnos) {
    for (const auto &anno : annos) {
      auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
      if (!sym) {
        newAnnos.push_back(anno.getAttr());
        continue;
      }

      // If not found, remove it!
      // This can happen if it was invalid in the input (mentions NLA that
      // didn't exist).
      auto it = nlaPrepass.origToVNLAs.find(sym.getAttr());
      if (it == nlaPrepass.origToVNLAs.end())
        continue;

      // Otherwise preserve it, cloning / modifying per active VNLAs.
      for (auto *matched : it->second) {
        auto path = matched->getPath();
        // If NLA was made local (path length zero), it now lives elsewhere.
        // If it wasn't local and we're not the destination, it lives elsewhere.
        // In both cases, drop this annotation.
        if (path.empty() || matched->getPath().back().finalMod != modName)
          continue;

        // If NLA is local otherwise, it originally had a 1-hop path.
        // Replace with local annotation.
        if (matched->isLocal()) {
          Annotation copy(anno);
          copy.removeMember("circt.nonlocal");
          newAnnos.push_back(copy.getAttr());
          continue;
        }

        // TODO: Unconditionally set in this loop? Local is "used".
        matched->wasUsed = true;

        // Keep as-is if we're using the same symbol as original.
        if (matched->realizedSym == matched->origSym) {
          assert(llvm::hasSingleElement(it->second));
          newAnnos.push_back(anno.getAttr());
          continue;
        }
        // Otherwise modify to use new symbol.
        Annotation copy(anno);
        copy.setMember("circt.nonlocal",
                       FlatSymbolRefAttr::get(matched->realizedSym));
        newAnnos.push_back(copy.getAttr());
      }
    }
  };

  // Update all annotations in circuit in parallel.
  auto fmodules =
      llvm::to_vector(circuit.getBodyBlock()->getOps<FModuleLike>());
  mlir::parallelForEach(context, fmodules, [&](FModuleLike fmodule) {
    SmallVector<Attribute> newAnnotations;
    fmodule.walk([&](Operation *op) {
      AnnotationSet annotations(op);
      // Early exit to avoid adding an empty annotations attribute to operations
      // which did not previously have annotations.
      if (annotations.empty())
        return;

      // Update annotations on the op.
      newAnnotations.clear();
      rewriteAnnos(annotations, fmodule.getModuleNameAttr(), newAnnotations);
      // annotations.removeAnnotations(processNLAs);
      // annotations.addAnnotations(newAnnotations);
      if (!newAnnotations.empty() || !annotations.empty())
        AnnotationSet(newAnnotations, context).applyToOperation(op);
    });

    // Update annotations on the ports.
    SmallVector<Attribute> newPortAnnotations;
    for (auto port : fmodule.getPorts()) {
      newAnnotations.clear();
      rewriteAnnos(port.annotations, fmodule.getModuleNameAttr(),
                   newAnnotations);
      newPortAnnotations.push_back(ArrayAttr::get(
          context, AnnotationSet(newAnnotations, context).getArray()));
    }
    fmodule->setAttr("portAnnotations",
                     ArrayAttr::get(context, newPortAnnotations));
  });

  // NLA cleanup and writeback
  auto b = OpBuilder::atBlockEnd(circuit.getBodyBlock());
  DenseMap<StringAttr, hw::HierPathOp> existingPaths;
  // TODO: Surely we have this information already earlier and can us that!
  for (auto nla : circuit.getBodyBlock()->getOps<hw::HierPathOp>())
    existingPaths[nla.getNameAttr()] = nla;

  for (auto &[_, vnlas] : nlaPrepass.origToVNLAs) {
    for (auto *vnla : vnlas) {
      if (!vnla->wasUsed || vnla->isLocal())
        continue;

      SmallVector<Attribute> pathAttrs;
      for (auto &hop : vnla->getPath()) {
        if (hop.finalInst)
          pathAttrs.push_back(InnerRefAttr::get(hop.finalMod, hop.finalInst));
        else
          pathAttrs.push_back(FlatSymbolRefAttr::get(hop.finalMod));
      }

      auto arrayAttr = b.getArrayAttr(pathAttrs);
      if (vnla->realizedSym == vnla->origSym) {
        // XXX: Not like this!
        existingPaths[vnla->origSym]->setAttr("namepath", arrayAttr);
        existingPaths.erase(vnla->origSym);
        continue;
      }

      auto hp = hw::HierPathOp::create(b, b.getUnknownLoc(), vnla->realizedSym,
                                       arrayAttr);
      hp.setPrivate();
    }
  }
  for (auto &[_, deadPath] : existingPaths)
    deadPath.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class InlinerPass : public circt::firrtl::impl::InlinerBase<InlinerPass> {
  void runOnOperation() override {
    CIRCT_DEBUG_SCOPED_PASS_LOGGER(this);
    auto circuit = getOperation();
    auto &symbolTable = getAnalysis<SymbolTable>();
    auto &instanceGraph = getAnalysis<InstanceGraph>();

    // Run classification prepass
    ClassificationPrepass classificationPrepass(circuit, instanceGraph,
                                                symbolTable);
    if (failed(classificationPrepass.run())) {
      signalPassFailure();
      return;
    }
    classificationPrepass.dump();

    // Run NLA prepass
    CircuitNamespace circuitNamespace(circuit);
    ModNamespaces modNamespaces;
    NLAPrepass nlaPrepass(circuit, circuitNamespace, symbolTable, instanceGraph,
                          modNamespaces, classificationPrepass.getModInfoMap());
    if (failed(nlaPrepass.run())) {
      signalPassFailure();
      return;
    }
    nlaPrepass.dump();

    LLVM_DEBUG({
      llvm::dbgs() << "\n=== Dumping circuit ===\n";
      circuit.dump();
    });

    Inliner inliner(circuit, symbolTable, instanceGraph, circuitNamespace,
                    modNamespaces, classificationPrepass, nlaPrepass);
    if (failed(inliner.run()))
      signalPassFailure();
  }
};
} // namespace
