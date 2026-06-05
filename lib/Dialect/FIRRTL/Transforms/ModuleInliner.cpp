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
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Support/Debug.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

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
using llvm::BitVector;

using InnerRefToNewNameMap = DenseMap<hw::InnerRefAttr, StringAttr>;

//===----------------------------------------------------------------------===//
// Module Inlining Support
//===----------------------------------------------------------------------===//

namespace {
/// Per-instantiation-context state for a non-local annotation being rewritten.
struct NLAContext {
  /// Sym name of the output `hw.hierpath` for this context.
  StringAttr outputSym;

  /// Root module name for this context.
  StringAttr root;

  /// Renamed inner-syms keyed by (post-inlining) module name.
  DenseMap<Attribute, StringAttr> renames;

  NLAContext(StringAttr outputSym, StringAttr root)
      : outputSym(outputSym), root(root), renames() {}
};

/// A representation of a source `hw.hierpath` (non-local annotation) that
/// can be mutated, then written back to the IR as one or more output
/// `hw.hierpath`s.
///
/// One `MutableNLA` is constructed per source `hw.hierpath` op in the
/// circuit and holds:
///   * Path structure (immutable across contexts): `nla`, `symIdx`, `size`.
///     `nla` is also the IR insertion anchor and what gets erased on
///     writeback.
///   * Path-level mutations (shared across contexts): `inlinedSymbols`,
///     `dead`, `moduleOnly`, `rootSet`.
///   * One or more per-context entries (`contexts`).  `contexts[0]` is
///     created at construction with the source NLA's identity (its
///     `outputSym` is the source sym; its `root` is the source root).
///     `reTop` either repurposes this default context (the first call) or
///     appends a fresh context with a freshly allocated sym (subsequent
///     calls).  `contexts[0]` keeping the source sym means a leaf
///     annotation referencing the source sym remains valid after writeback.
///
/// Lifecycle:
///   1. Constructed from a source `hw.hierpath` op in `Inliner::run()`.
///   2. `inlineModule`/`flattenModule`/`reTop`/`setInnerSym` accumulate
///      mutations during inlining.
///   3. `applyUpdates` emits one new `hw.hierpath` per context and erases
///      the source op, or leaves it unchanged if nothing changed.
class MutableNLA {
  /// Source `hw.hierpath` op.  Also the IR insertion anchor for emitted
  /// outputs; erased by `applyUpdates` when writeback occurs.
  hw::HierPathOp nla;

  /// A namespace that can be used to generate new symbol names if needed.
  CircuitNamespace *circuitNamespace;

  /// Maps path module name to index in the source path.
  DenseMap<Attribute, unsigned> symIdx;

  /// Which path elements are still present.  True = module remains in path;
  /// false = inlined/flattened away.
  BitVector inlinedSymbols;

  /// True if the NLA has no live targets (leaf was inlined/flattened away).
  bool dead = false;

  /// True if the NLA targets only a module, not a port or op within it.
  bool moduleOnly = false;

  /// All modules at which this NLA is (or has been) rooted.
  DenseSet<StringAttr> rootSet;

  /// Length of the source NLA's path.
  unsigned int size;

  /// One entry per output `hw.hierpath` to emit.
  SmallVector<NLAContext> contexts;

  /// True after the first `reTop` call.
  bool retopped = false;

  /// Look up the inner-sym to use for path step `idx` in context `ctx`.
  /// Uses the context's renames; falls back to the source NLA's refPart.
  StringAttr lookupRename(const NLAContext &ctx, Attribute lastMod,
                          unsigned idx) {
    auto it = ctx.renames.find(lastMod);
    if (it != ctx.renames.end())
      return it->second;
    return nla.refPart(idx);
  }

public:
  MutableNLA(hw::HierPathOp nla, CircuitNamespace *circuitNamespace)
      : nla(nla), circuitNamespace(circuitNamespace),
        inlinedSymbols(BitVector(nla.getNamepath().size(), true)),
        size(nla.getNamepath().size()) {
    for (size_t i = 0, e = size; i != e; ++i)
      symIdx.insert({nla.modPart(i), i});
    rootSet.insert(nla.root());
    // Default context inherits the source NLA's identity.
    contexts.push_back({nla.getSymNameAttr(), nla.root()});
  }

  void markDead() { dead = true; }

  bool isRetopped() const { return retopped; }

  /// Mark the NLA as only used to target a module (no ports/ops use it).
  void markModuleOnly() { moduleOnly = true; }

  /// Return the source `hw.hierpath` op.
  hw::HierPathOp getNLA() { return nla; }

  /// Writeback updates accumulated in this MutableNLA: emit one new
  /// `hw.hierpath` op per `NLAContext`, then erase the source op.
  ///
  /// Fast path: if nothing changed (not retop'd, no inlining, no renames),
  /// the source op is left in place unchanged.
  ///
  /// This method should only ever be called once.  After calling this,
  /// the MutableNLA must not be used further.
  void applyUpdates() {
    // Fast path: source op is unchanged, leave it in place.
    if (!dead && !retopped && inlinedSymbols.all() &&
        contexts[0].renames.empty()) {
      assert(contexts.size() == 1);
      return;
    }
    OpBuilder b(nla);
    if (!dead)
      for (auto &ctx : contexts)
        writeBackContext(b, ctx);
    nla.erase();
  }

private:
  /// Build and emit one output `hw.hierpath` for `ctx`, applying inline
  /// state and the context's renames to the source path.
  hw::HierPathOp writeBackContext(OpBuilder &b, const NLAContext &ctx) {
    SmallVector<Attribute> namepath;
    StringAttr lastMod;

    // Root of the namepath.  If the next module has been inlined, set
    // lastMod to root and skip adding to the namepath.  Otherwise, add
    // the root with its inner ref.
    if (!inlinedSymbols.test(1)) {
      lastMod = ctx.root;
    } else {
      namepath.push_back(
          InnerRefAttr::get(ctx.root, lookupRename(ctx, ctx.root, 0)));
    }

    // Middle of the namepath (excluding root and leaf).
    for (signed i = 1, e = inlinedSymbols.size() - 1; i != e; ++i) {
      if (!inlinedSymbols.test(i + 1)) {
        if (!lastMod)
          lastMod = nla.modPart(i);
        continue;
      }
      auto modPart = lastMod ? lastMod : nla.modPart(i);
      auto refPart = lookupRename(ctx, modPart, i);
      namepath.push_back(InnerRefAttr::get(modPart, refPart));
      lastMod = {};
    }

    // Leaf.
    auto modPart = lastMod ? lastMod : nla.modPart(size - 1);
    auto refPart = lookupRename(ctx, modPart, size - 1);
    if (refPart)
      namepath.push_back(InnerRefAttr::get(modPart, refPart));
    else
      namepath.push_back(FlatSymbolRefAttr::get(modPart));

    auto hp = hw::HierPathOp::create(b, b.getUnknownLoc(), ctx.outputSym,
                                     b.getArrayAttr(namepath));
    hp.setVisibility(nla.getVisibility());
    return hp;
  }

public:

  void dump() {
    llvm::errs() << "  - orig:           " << nla << "\n"
                 << "    new:            " << *this << "\n"
                 << "    dead:           " << dead << "\n"
                 << "    isDead:         " << isDead() << "\n"
                 << "    isModuleOnly:   " << isModuleOnly() << "\n"
                 << "    isLocal:        " << isLocal() << "\n"
                 << "    inlinedSymbols: [";
    llvm::interleaveComma(inlinedSymbols.getData(), llvm::errs(), [](auto a) {
      llvm::errs() << llvm::formatv("{0:x-}", a);
    });
    llvm::errs() << "]\n"
                 << "    contexts:\n";
    for (auto &ctx : contexts) {
      llvm::errs() << "      - outputSym: " << ctx.outputSym << "\n"
                   << "        root:      " << ctx.root << "\n"
                   << "        renames:\n";
      for (auto &rn : ctx.renames)
        llvm::errs() << "          - " << rn.first << " -> " << rn.second
                     << "\n";
    }
  }

  /// Write the current state of this MutableNLA in a format that looks like
  /// NLA serialization.  Intended for debugging.
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, MutableNLA &x) {
    auto writePathSegment = [&](StringAttr mod, StringAttr sym = {}) {
      if (sym)
        os << "#hw.innerNameRef<";
      os << "@" << mod.getValue();
      if (sym)
        os << "::@" << sym.getValue() << ">";
    };

    auto writeOne = [&](const NLAContext &ctx) {
      os << "firrtl.nla @" << ctx.outputSym.getValue() << " [";
      StringAttr lastMod;
      bool needsComma = false;
      auto lookup = [&](Attribute mod, unsigned idx) -> StringAttr {
        auto it = ctx.renames.find(mod);
        if (it != ctx.renames.end())
          return it->second;
        return x.nla.refPart(idx);
      };

      if (!x.inlinedSymbols.test(1)) {
        lastMod = ctx.root;
      } else {
        writePathSegment(ctx.root, lookup(ctx.root, 0));
        needsComma = true;
      }
      for (signed i = 1, e = x.inlinedSymbols.size() - 1; i != e; ++i) {
        if (!x.inlinedSymbols.test(i + 1)) {
          if (!lastMod)
            lastMod = x.nla.modPart(i);
          continue;
        }
        if (needsComma)
          os << ", ";
        auto modPart = lastMod ? lastMod : x.nla.modPart(i);
        writePathSegment(modPart, lookup(modPart, i));
        needsComma = true;
        lastMod = {};
      }
      if (needsComma)
        os << ", ";
      auto modPart = lastMod ? lastMod : x.nla.modPart(x.size - 1);
      writePathSegment(modPart, lookup(modPart, x.size - 1));
      os << "]";
    };

    bool multiary = x.contexts.size() > 1;
    if (multiary)
      os << "[";
    llvm::interleaveComma(x.contexts, os,
                          [&](const NLAContext &c) { writeOne(c); });
    if (multiary)
      os << "]";
    return os;
  }

  /// Returns true if this NLA is dead.  Set by `markDead` when its
  /// module-only leaf was flattened or when its only target was inlined
  /// away.  When dead, `applyUpdates` writes no outputs.
  bool isDead() { return dead; }

  /// Returns true if this NLA targets only a module.
  bool isModuleOnly() { return moduleOnly; }

  /// Returns true if this NLA is local.  For this to be local, every module
  /// after the root must be inlined.  The root is never truly inlined as
  /// inlining the root just sets a new root.
  bool isLocal() {
    return inlinedSymbols.find_first_in(1, inlinedSymbols.size()) == -1;
  }

  /// Return true if this NLA has a root that originates from a specific module.
  bool hasRoot(FModuleLike mod) {
    return (isDead() && nla.root() == mod.getModuleNameAttr()) ||
           rootSet.contains(mod.getModuleNameAttr());
  }

  /// Return true if either this NLA is rooted at modName, or is retoped to it.
  bool hasRoot(StringAttr modName) {
    return (nla.root() == modName) || rootSet.contains(modName);
  }

  /// Mark a module as inlined.  This will remove it from the NLA.
  void inlineModule(FModuleOp module) {
    auto sym = module.getNameAttr();
    assert(sym != nla.root() && "unable to inline the root module");
    assert(symIdx.count(sym) && "module is not in the symIdx map");
    auto idx = symIdx[sym];
    inlinedSymbols.reset(idx);
    // If we inlined the last module in the path and the NLA targets only that
    // module, then this NLA is dead.
    if (idx == size - 1 && moduleOnly)
      markDead();
  }

  /// Mark a module as flattened.  This has the effect of inlining all of its
  /// children.  Also mark the NLA as dead if the leaf reference of this NLA is
  /// a module and the only target is a module.
  void flattenModule(FModuleOp module) {
    auto sym = module.getNameAttr();
    assert(symIdx.count(sym) && "module is not in the symIdx map");
    // When flattening a module, all modules from this point onwards in the path
    // are effectively inlined. Reset all bits from the module's index onwards.
    auto moduleIdx = symIdx[sym];
    inlinedSymbols.reset(moduleIdx, size);
    // If the NLA only targets a module and we're flattening the NLA,
    // then the NLA must be dead.  Mark it as such.
    if (moduleOnly)
      markDead();
  }

  /// Re-root the NLA at `module`.  Returns the output sym for this context.
  /// The first call reuses the source sym (so existing leaf annotations stay
  /// valid); subsequent calls allocate a fresh sym.
  StringAttr reTop(FModuleOp module) {
    StringAttr modName = module.getNameAttr();
    StringAttr newSym;
    if (!retopped) {
      contexts[0].root = modName;
      newSym = contexts[0].outputSym;
      retopped = true;
    } else {
      StringAttr sourceSym = nla.getSymNameAttr();
      newSym = StringAttr::get(nla.getContext(),
                               circuitNamespace->newName(sourceSym.getValue()));
      contexts.push_back({newSym, modName});
    }
    rootSet.insert(modName);
    symIdx.insert({modName, 0});
    return newSym;
  }

  /// Output syms for all instantiation contexts, in creation order.
  SmallVector<StringAttr> getOutputSyms() const {
    SmallVector<StringAttr> syms;
    syms.reserve(contexts.size());
    for (auto &ctx : contexts)
      syms.push_back(ctx.outputSym);
    return syms;
  }

  /// Record a renamed inner-symbol for `module` on the context identified by
  /// `outputSym`.
  void setInnerSym(StringAttr outputSym, Attribute module,
                   StringAttr innerSym) {
    assert(symIdx.count(module) && "module not in this NLA's path");
    NLAContext *ctx = findContext(outputSym);
    assert(ctx && "setInnerSym called with unknown outputSym");
    assert(!ctx->renames.count(module) && "Module already renamed");
    ctx->renames.insert({module, innerSym});
  }

  /// Find a context by output sym.  Returns null if not present.
  NLAContext *findContext(StringAttr outputSym) {
    for (auto &ctx : contexts)
      if (ctx.outputSym == outputSym)
        return &ctx;
    return nullptr;
  }
};
} // namespace

/// This function is used after inlining a module, to handle the conversion
/// between module ports and instance results. This maps each wire to the
/// result of the instance operation.  When future operations are cloned from
/// the current block, they will use the value of the wire instead of the
/// instance results.
static void mapResultsToWires(IRMapping &mapper, SmallVectorImpl<Value> &wires,
                              InstanceOp instance) {
  for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i) {
    auto result = instance.getResult(i);
    auto wire = wires[i];
    mapper.map(result, wire);
  }
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
/// The inliner works in a top down fashion, starting from the top level module,
/// and inlines every possible instance. With this method of recursive top-down
/// inlining, each operation will be cloned directly to its final location.
///
/// The inliner uses a worklist to track which modules need to be processed.
/// When an instance op is not inlined, the referenced module is added to the
/// worklist. When the inliner is complete, it deletes every un-processed
/// module: either all instances of the module were inlined, or it was not
/// reachable from the top level module.
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
  Inliner(CircuitOp circuit, SymbolTable &symbolTable);

  /// Run the inliner.
  LogicalResult run();

private:
  /// Inlining context, one per module being inlined into.
  /// Cleans up backedges on destruction.
  struct ModuleInliningContext {
    ModuleInliningContext(FModuleOp module)
        : module(module), modNamespace(module), b(module.getContext()) {}
    /// Top-level module for current inlining task.
    FModuleOp module;
    /// Namespace for generating new names in `module`.
    hw::InnerSymbolNamespace modNamespace;
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

    ~InliningLevel() {
      replaceInnerRefUsers(newOps, relocatedInnerSyms,
                           mic.module.getNameAttr());
    }
  };

  /// Return the active output sym for the MutableNLA identified by
  /// `annotationSym` — the output sym that is present in `activeHierpaths`.
  /// Returns a null StringAttr if no context is active on the current path.
  /// Handles retop'd NLAs where the annotation sym (source sym / context 0)
  /// differs from the active context's sym.
  StringAttr findActiveOutputSym(StringAttr annotationSym);

  /// Rename an operation and unique any symbols it has.
  /// Returns true iff symbol was changed.
  bool rename(StringRef prefix, Operation *op, InliningLevel &il);

  /// Rename an InstanceOp and unique any symbols it has.
  /// Requires old and new operations to appropriately update the `HierPathOp`'s
  /// that it participates in.
  bool renameInstance(StringRef prefix, InliningLevel &il, InstanceOp oldInst,
                      InstanceOp newInst,
                      const DenseMap<Attribute, Attribute> &symbolRenames);

  /// Clone and rename an operation.  Insert the operation into the inlining
  /// level.
  void cloneAndRename(StringRef prefix, InliningLevel &il, IRMapping &mapper,
                      Operation &op,
                      const DenseMap<Attribute, Attribute> &symbolRenames,
                      const DenseSet<Attribute> &localSymbols);

  /// Rewrite the ports of a module as wires.  This is similar to
  /// cloneAndRename, but operating on ports.
  /// Wires are added to il.wires.
  void mapPortsToWires(StringRef prefix, InliningLevel &il, IRMapping &mapper,
                       const DenseSet<Attribute> &localSymbols);

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
                            IRMapping &mapper,
                            DenseSet<Attribute> localSymbols);

  /// Inlines a target module into the insertion point of the builder,
  /// prefixing all operations with prefix.  This clones all operations from
  /// the target, and does not trigger inlining on the target itself.
  LogicalResult inlineInto(StringRef prefix, InliningLevel &il,
                           IRMapping &mapper,
                           DenseMap<Attribute, Attribute> &symbolRenames);

  /// Recursively flatten all instances in a module.
  LogicalResult flattenInstances(FModuleOp module);

  /// Inline any instances in the module which were marked for inlining.
  LogicalResult inlineInstances(FModuleOp module);

  /// Create a debug scope for an inlined instance at the current insertion
  /// point of the `il.mic` builder.
  void createDebugScope(InliningLevel &il, InstanceOp instance,
                        Value parentScope = {});

  /// Identify all module-only NLA's, marking their MutableNLA's accordingly.
  void identifyNLAsTargetingOnlyModules();

  /// Mark referenced modules of an unknown FInstanceLike operation as live.
  /// For non-InstanceOp FInstanceLike operations (e.g., InstanceChoiceOp,
  /// ObjectOp), we cannot inline them, so we need to mark their referenced
  /// modules as live to prevent them from being deleted.
  void markUnknownFInstanceLikeModulesLive(FInstanceLike instanceLike) {
    for (auto module : instanceLike.getReferencedModuleNamesAttr()
                           .getAsValueRange<StringAttr>()) {
      auto *moduleOp = symbolTable.lookup(module);
      if (liveModules.insert(moduleOp).second) {
        if (auto fmodule = dyn_cast<FModuleOp>(moduleOp))
          worklist.push_back(fmodule);
      }
    }
  }

  /// Populate the activeHierpaths with the HierPaths that are active given the
  /// current hierarchy. This is the set of HierPaths that were active in the
  /// parent, and on the current instance. Also HierPaths that are rooted at
  /// this module are also added to the active set.
  void setActiveHierPaths(StringAttr moduleName, StringAttr instInnerSym) {
    auto &instPaths =
        instOpHierPaths[InnerRefAttr::get(moduleName, instInnerSym)];
    if (currentPath.empty()) {
      activeHierpaths.insert(instPaths.begin(), instPaths.end());
      return;
    }
    DenseSet<StringAttr> hPaths(instPaths.begin(), instPaths.end());
    // Intersect with the parent active set, preserving per-context output syms:
    // a retop'd sym not literally in instPaths still belongs if its MutableNLA
    // is represented there under a different output sym.
    auto parent = activeHierpaths;
    activeHierpaths.clear();
    for (auto sym : parent) {
      if (hPaths.contains(sym)) {
        activeHierpaths.insert(sym);
        continue;
      }
      // sym is a per-context output sym; check whether any hPath shares its
      // MutableNLA.
      for (auto h : hPaths) {
        auto it = nlaMap.find(h);
        if (it == nlaMap.end())
          continue;
        for (auto outSym : it->second->getOutputSyms())
          if (outSym == sym) {
            activeHierpaths.insert(sym);
            break;
          }
        if (activeHierpaths.contains(sym))
          break;
      }
    }
    // Add NLAs rooted at this instance.  For retop'd NLAs, prefer the output
    // sym already active in the parent path; fall back to the source sym.
    for (auto hPath : instPaths) {
      auto it = nlaMap.find(hPath);
      if (it == nlaMap.end())
        continue;
      if (!it->second->hasRoot(moduleName))
        continue;
      StringAttr toAdd = hPath;
      for (auto outSym : it->second->getOutputSyms())
        if (parent.contains(outSym)) {
          toAdd = outSym;
          break;
        }
      activeHierpaths.insert(toAdd);
    }
  }

  CircuitOp circuit;
  MLIRContext *context;

  // A symbol table with references to each module in a circuit.
  SymbolTable &symbolTable;

  /// The set of live modules.  Anything not recorded in this set will be
  /// removed by dead code elimination.
  DenseSet<Operation *> liveModules;

  /// Worklist of modules to process for inlining or flattening.
  SmallVector<FModuleOp, 16> worklist;

  /// Owns the `MutableNLA`s.  Reserved once up-front in `run()` to the
  /// number of source `hw.hierpath` ops in the circuit, then populated
  /// without further reallocation, so pointers into this vector are
  /// stable for the duration of the pass.
  SmallVector<MutableNLA> nlaStorage;

  /// Maps output sym -> MutableNLA.  After reTop, multiple syms (one per
  /// instantiation context) can map to the same MutableNLA.
  DenseMap<Attribute, MutableNLA *> nlaMap;

  /// Source NLA syms in IR order, for deterministic writeback iteration.
  SmallVector<StringAttr> sourceSymsInOrder;

  /// A mapping of module names to NLA symbols that originate from that module.
  DenseMap<Attribute, SmallVector<Attribute>> rootMap;

  /// The current instance path.  This is a pair<ModuleName, InstanceName>.
  /// This is used to distinguish if a non-local annotation applies to the
  /// current instance or not.
  SmallVector<std::pair<Attribute, Attribute>> currentPath;

  DenseSet<StringAttr> activeHierpaths;

  /// Record the HierPathOps that each InstanceOp participates in. This is a map
  /// from the InnerRefAttr to the list of HierPathOp names. The InnerRefAttr
  /// corresponds to the InstanceOp.
  DenseMap<InnerRefAttr, SmallVector<StringAttr>> instOpHierPaths;

  /// The debug scopes created for inlined instances. Scopes that are unused
  /// after inlining will be deleted again.
  SmallVector<debug::ScopeOp> debugScopes;
};
} // namespace

/// Return the output sym of the active context for the given annotation sym,
/// or a null StringAttr if no context is active on the current path.
///
/// Annotations reference the source NLA sym (which equals context 0's output
/// sym).  For retop'd NLAs the active context may be a different context whose
/// output sym was freshly allocated.  This function searches all output syms
/// so it works regardless of whether the NLA has been retop'd.
StringAttr Inliner::findActiveOutputSym(StringAttr annotationSym) {
  auto it = nlaMap.find(annotationSym);
  if (it == nlaMap.end())
    return {};
  for (auto outSym : it->second->getOutputSyms())
    if (activeHierpaths.count(outSym))
      return outSym;
  return {};
}

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

  // If there's a symbol on the root and it changed, do NLA work.
  if (auto newSymStrAttr = newSymAttr.getSymName();
      newSymStrAttr && newSymStrAttr != oldSymAttr.getSymName()) {
    for (Annotation anno : AnnotationSet(op)) {
      auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
      if (!sym)
        continue;
      auto *mnla = nlaMap[sym.getAttr()];
      auto activeSym = findActiveOutputSym(sym.getAttr());
      if (!activeSym)
        continue;
      mnla->setInnerSym(activeSym, il.mic.module.getModuleNameAttr(),
                        newSymStrAttr);
    }
  }

  symOp.setInnerSymbolAttr(newSymAttr);

  return newSymAttr != oldSymAttr;
}

bool Inliner::renameInstance(
    StringRef prefix, InliningLevel &il, InstanceOp oldInst, InstanceOp newInst,
    const DenseMap<Attribute, Attribute> &symbolRenames) {
  // TODO: There is currently no good way to annotate an explicit parent scope
  // on instances. Just emit a note in debug runs until this is resolved.
  LLVM_DEBUG({
    if (il.debugScope)
      llvm::dbgs() << "Discarding parent debug scope for " << oldInst << "\n";
  });

  // Add this instance to the activeHierpaths. This ensures that NLAs that this
  // instance participates in will be updated correctly.
  auto parentActivePaths = activeHierpaths;
  assert(oldInst->getParentOfType<FModuleOp>() == il.childModule);
  if (auto instSym = getInnerSymName(oldInst))
    setActiveHierPaths(oldInst->getParentOfType<FModuleOp>().getNameAttr(),
                       instSym);
  // List of HierPathOps that are valid based on the InstanceOp being inlined
  // and the InstanceOp which is being replaced after inlining. That is the set
  // of HierPathOps that is common between these two.
  SmallVector<StringAttr> validHierPaths;
  auto oldParent = oldInst->getParentOfType<FModuleOp>().getNameAttr();
  auto oldInstSym = getInnerSymName(oldInst);

  if (oldInstSym) {
    // Get the innerRef to the original InstanceOp that is being inlined here.
    // For all the HierPathOps that the instance being inlined participates
    // in.
    auto oldInnerRef = InnerRefAttr::get(oldParent, oldInstSym);
    for (auto old : instOpHierPaths[oldInnerRef]) {
      // If this HierPathOp is valid at the inlining context, where the
      // instance is being inlined at. That is, if it exists in the
      // activeHierpaths.
      if (activeHierpaths.find(old) != activeHierpaths.end())
        validHierPaths.push_back(old);
      else
        // The HierPathOp could have been renamed, check for the other retop
        // output syms.  Push the retop'd output sym so subsequent
        // setInnerSym lands on the right per-context state.
        for (auto outSym : nlaMap[old]->getOutputSyms())
          if (activeHierpaths.find(outSym) != activeHierpaths.end()) {
            validHierPaths.push_back(outSym);
            break;
          }
    }
  }

  assert(getInnerSymName(newInst) == oldInstSym);

  // Do the renaming, creating new symbol as needed.
  auto symbolChanged = rename(prefix, newInst, il);

  // If the symbol changed, update instOpHierPaths accordingly.
  auto newSymAttr = getInnerSymName(newInst);
  if (symbolChanged) {
    assert(newSymAttr);
    // The InstanceOp is renamed, so move the HierPathOps to the new
    // InnerRefAttr.
    auto newInnerRef = InnerRefAttr::get(
        newInst->getParentOfType<FModuleOp>().getNameAttr(), newSymAttr);
    instOpHierPaths[newInnerRef] = validHierPaths;
    // Update the innerSym for all the affected HierPathOps.
    for (auto outSym : instOpHierPaths[newInnerRef]) {
      auto it = nlaMap.find(outSym);
      if (it == nlaMap.end())
        continue;
      it->second->setInnerSym(outSym, newInnerRef.getModule(), newSymAttr);
    }
  }

  if (newSymAttr) {
    auto innerRef = InnerRefAttr::get(
        newInst->getParentOfType<FModuleOp>().getNameAttr(), newSymAttr);
    SmallVector<StringAttr> &nlaList = instOpHierPaths[innerRef];
    // Now rename the Updated HierPathOps that this InstanceOp participates in.
    for (const auto &en : llvm::enumerate(nlaList)) {
      auto oldNLA = en.value();
      if (auto newSym = symbolRenames.lookup(oldNLA))
        nlaList[en.index()] = cast<StringAttr>(newSym);
    }
  }
  activeHierpaths = std::move(parentActivePaths);
  return symbolChanged;
}

/// This function is used before inlining a module, to handle the conversion
/// between module ports and instance results. For every port in the target
/// module, create a wire, and assign a mapping from each module port to the
/// wire. When the body of the module is cloned, the value of the wire will be
/// used instead of the module's ports.
void Inliner::mapPortsToWires(StringRef prefix, InliningLevel &il,
                              IRMapping &mapper,
                              const DenseSet<Attribute> &localSymbols) {
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

    StringAttr newRootSymName, oldRootSymName;
    if (oldSymAttr)
      oldRootSymName = oldSymAttr.getSymName();
    if (newSymAttr)
      newRootSymName = newSymAttr.getSymName();

    SmallVector<Attribute> newAnnotations;
    for (auto anno : AnnotationSet::forPort(target, i)) {
      // If the annotation is not non-local, copy it to the clone.
      if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
        auto *mnla = nlaMap[sym.getAttr()];
        auto activeSym = findActiveOutputSym(sym.getAttr());
        if (!activeSym)
          continue;
        if (oldRootSymName != newRootSymName)
          mnla->setInnerSym(activeSym,
                            il.mic.module.getModuleNameAttr(), newRootSymName);
        if (mnla->isLocal() || localSymbols.count(activeSym))
          anno.removeMember("circt.nonlocal");
        else if (activeSym != sym.getAttr())
          anno.setMember("circt.nonlocal", FlatSymbolRefAttr::get(activeSym));
      }
      newAnnotations.push_back(anno.getAttr());
    }

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
void Inliner::cloneAndRename(
    StringRef prefix, InliningLevel &il, IRMapping &mapper, Operation &op,
    const DenseMap<Attribute, Attribute> &symbolRenames,
    const DenseSet<Attribute> &localSymbols) {
  // Strip any non-local annotations which are local.
  AnnotationSet oldAnnotations(&op);
  SmallVector<Annotation> newAnnotations;
  for (auto anno : oldAnnotations) {
    // If the annotation is not non-local, it will apply to all inlined
    // instances of this op. Add it to the cloned op.
    if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
      auto *mnla = nlaMap[sym.getAttr()];
      auto activeSym = findActiveOutputSym(sym.getAttr());
      if (!activeSym)
        continue;
      if (mnla->isLocal() || localSymbols.count(activeSym))
        anno.removeMember("circt.nonlocal");
      else if (activeSym != sym.getAttr())
        anno.setMember("circt.nonlocal", FlatSymbolRefAttr::get(activeSym));
    }
    // Attach this annotation to the cloned operation.
    newAnnotations.push_back(anno);
  }

  // Clone and rename.
  assert(op.getNumRegions() == 0 &&
         "operation with regions should not reach cloneAndRename");
  auto *newOp = il.mic.b.cloneWithoutRegions(op, mapper);

  // Rename the new operation.
  // (add prefix to it, if named, and unique-ify symbol, updating NLA's).

  // Instances require extra handling to update HierPathOp's if their symbols
  // change.
  if (auto oldInst = dyn_cast<InstanceOp>(op))
    renameInstance(prefix, il, oldInst, cast<InstanceOp>(newOp), symbolRenames);
  else
    rename(prefix, newOp, il);

  // We want to avoid attaching an empty annotation array on to an op that
  // never had an annotation array in the first place.
  if (!newAnnotations.empty() || !oldAnnotations.empty())
    AnnotationSet(newAnnotations, context).applyToOperation(newOp);

  il.newOps.push_back(newOp);
}

bool Inliner::shouldFlatten(Operation *op) {
  return AnnotationSet::hasAnnotation(op, flattenAnnoClass);
}

bool Inliner::shouldInline(Operation *op) {
  return AnnotationSet::hasAnnotation(op, inlineAnnoClass);
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
                                   IRMapping &mapper,
                                   DenseSet<Attribute> localSymbols) {
  auto target = il.childModule;
  auto moduleName = target.getNameAttr();
  DenseMap<Attribute, Attribute> symbolRenames;

  LLVM_DEBUG(llvm::dbgs() << "flattening " << target.getModuleName() << " into "
                          << il.mic.module.getModuleName() << "\n");
  auto visit = [&](Operation *op) {
    // If it's not an instance op, clone it and continue.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance) {
      if (auto instanceLike = dyn_cast<FInstanceLike>(op))
        markUnknownFInstanceLikeModulesLive(instanceLike);
      cloneAndRename(prefix, il, mapper, *op, symbolRenames, localSymbols);
      return success();
    }

    // If it's not a regular module we can't inline it. Mark it as live.
    auto *moduleOp = symbolTable.lookup(instance.getModuleName());
    auto childModule = dyn_cast<FModuleOp>(moduleOp);
    if (!childModule) {
      liveModules.insert(moduleOp);

      cloneAndRename(prefix, il, mapper, *op, symbolRenames, localSymbols);
      return success();
    }

    if (failed(checkInstanceParents(instance)))
      return failure();

    // Add any NLAs which start at this instance to the localSymbols set.
    // Anything in this set will be made local during the recursive flattenInto
    // walk.
    llvm::set_union(localSymbols, rootMap[childModule.getNameAttr()]);
    auto instInnerSym = getInnerSymName(instance);
    auto parentActivePaths = activeHierpaths;
    setActiveHierPaths(moduleName, instInnerSym);
    currentPath.emplace_back(moduleName, instInnerSym);

    InliningLevel childIL(il.mic, childModule);
    createDebugScope(childIL, instance, il.debugScope);

    // Create the wire mapping for results + ports.
    auto nestedPrefix = (prefix + instance.getName() + "_").str();
    mapPortsToWires(nestedPrefix, childIL, mapper, localSymbols);
    mapResultsToWires(mapper, childIL.wires, instance);

    // Unconditionally flatten all instance operations.
    if (failed(flattenInto(nestedPrefix, childIL, mapper, localSymbols)))
      return failure();
    currentPath.pop_back();
    activeHierpaths = parentActivePaths;
    return success();
  };
  return inliningWalk(il.mic.b, target.getBodyBlock(), mapper, visit);
}

LogicalResult Inliner::flattenInstances(FModuleOp module) {
  auto moduleName = module.getNameAttr();
  ModuleInliningContext mic(module);

  auto visit = [&](FInstanceLike instanceLike) {
    auto instance = dyn_cast<InstanceOp>(*instanceLike);
    if (!instance) {
      markUnknownFInstanceLikeModulesLive(instanceLike);
      return WalkResult::advance();
    }
    // If it's not a regular module we can't inline it. Mark it as live.
    auto *targetModule = symbolTable.lookup(instance.getModuleName());
    auto target = dyn_cast<FModuleOp>(targetModule);
    if (!target) {
      liveModules.insert(targetModule);
      return WalkResult::advance();
    }

    if (failed(checkInstanceParents(instance)))
      return WalkResult::interrupt();

    if (auto instSym = getInnerSymName(instance)) {
      auto innerRef = InnerRefAttr::get(moduleName, instSym);
      // Preorder update of any non-local annotations this instance participates
      // in.  This needs to happen _before_ visiting modules so that internal
      // non-local annotations can be deleted if they are now local.
      for (auto targetNLA : instOpHierPaths[innerRef])
        nlaMap[targetNLA]->flattenModule(target);
    }

    // Add any NLAs which start at this instance to the localSymbols set.
    // Anything in this set will be made local during the recursive flattenInto
    // walk.
    DenseSet<Attribute> localSymbols;
    llvm::set_union(localSymbols, rootMap[target.getNameAttr()]);
    auto instInnerSym = getInnerSymName(instance);
    auto parentActivePaths = activeHierpaths;
    setActiveHierPaths(moduleName, instInnerSym);
    currentPath.emplace_back(moduleName, instInnerSym);

    // Create the wire mapping for results + ports. We RAUW the results instead
    // of mapping them.
    IRMapping mapper;
    mic.b.setInsertionPoint(instance);

    InliningLevel il(mic, target);
    createDebugScope(il, instance);

    auto nestedPrefix = (instance.getName() + "_").str();
    mapPortsToWires(nestedPrefix, il, mapper, localSymbols);
    for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i)
      instance.getResult(i).replaceAllUsesWith(il.wires[i]);

    // Recursively flatten the target module.
    if (failed(flattenInto(nestedPrefix, il, mapper, localSymbols)))
      return WalkResult::interrupt();
    currentPath.pop_back();
    activeHierpaths = parentActivePaths;

    // Erase the replaced instance.
    instance.erase();
    return WalkResult::skip();
  };
  return failure(module.getBodyBlock()
                     ->walk<mlir::WalkOrder::PreOrder>(visit)
                     .wasInterrupted());
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult
Inliner::inlineInto(StringRef prefix, InliningLevel &il, IRMapping &mapper,
                    DenseMap<Attribute, Attribute> &symbolRenames) {
  auto target = il.childModule;
  auto inlineToParent = il.mic.module;
  auto moduleName = target.getNameAttr();

  LLVM_DEBUG(llvm::dbgs() << "inlining " << target.getModuleName() << " into "
                          << inlineToParent.getModuleName() << "\n");

  auto visit = [&](Operation *op) {
    // If it's not an instance op, clone it and continue.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance) {
      if (auto instanceLike = dyn_cast<FInstanceLike>(op))
        markUnknownFInstanceLikeModulesLive(instanceLike);

      cloneAndRename(prefix, il, mapper, *op, symbolRenames, {});
      return success();
    }

    // If it's not a regular module we can't inline it. Mark it as live.
    auto *moduleOp = symbolTable.lookup(instance.getModuleName());
    auto childModule = dyn_cast<FModuleOp>(moduleOp);
    if (!childModule) {
      liveModules.insert(moduleOp);
      cloneAndRename(prefix, il, mapper, *op, symbolRenames, {});
      return success();
    }

    // If we aren't inlining the target, add it to the work list.
    if (!shouldInline(childModule)) {
      if (liveModules.insert(childModule).second) {
        worklist.push_back(childModule);
      }
      cloneAndRename(prefix, il, mapper, *op, symbolRenames, {});
      return success();
    }

    if (failed(checkInstanceParents(instance)))
      return failure();

    auto toBeFlattened = shouldFlatten(childModule);
    if (auto instSym = getInnerSymName(instance)) {
      auto innerRef = InnerRefAttr::get(moduleName, instSym);
      // Preorder update of any non-local annotations this instance participates
      // in.  This needs to happen _before_ visiting modules so that internal
      // non-local annotations can be deleted if they are now local.
      for (auto sym : instOpHierPaths[innerRef]) {
        auto *mnla = nlaMap[sym];
        // Skip if this NLA is rooted at childModule: the root has been
        // retop'd already (or will be by the rootMap block below) and it is
        // an error to inlineModule on the root.  This guards against state
        // left over from a previous walk of the same module body (e.g., a
        // reTop on a prior instance added an inner sym + instOpHierPaths
        // entry that we now re-encounter).
        if (mnla->getNLA().root() == childModule.getNameAttr())
          continue;
        if (toBeFlattened)
          mnla->flattenModule(childModule);
        else
          mnla->inlineModule(childModule);
      }
    }

    // The InstanceOp `instance` might not have a symbol, if it does not
    // participate in any HierPathOp. But the reTop might add a symbol to it, if
    // a HierPathOp is added to this Op. If we're about to inline a module that
    // contains a non-local annotation that starts at that module, then we need
    // to both update the mutable NLA to indicate that this has a new top and
    // add an annotation on the instance saying that this now participates in
    // this new NLA.
    DenseMap<Attribute, Attribute> symbolRenames;
    if (!rootMap[childModule.getNameAttr()].empty()) {
      for (auto origSymAttr : rootMap[childModule.getNameAttr()]) {
        auto origSym = cast<StringAttr>(origSymAttr);
        auto *mnla = nlaMap[origSym];
        auto origNLAName = mnla->getNLA().getNameAttr();
        // Retop to the new parent, which is the topmost module (and not
        // immediate parent) in case of recursive inlining.  Returns the
        // (possibly fresh) outputSym for the resulting per-context entry.
        StringAttr newSym = mnla->reTop(inlineToParent);
        // If reTop allocated a fresh sym (a 2nd-or-later context), register
        // it in nlaMap so future lookups via newSym find the same MutableNLA.
        if (newSym != origSym)
          nlaMap[newSym] = mnla;
        StringAttr instSym = getInnerSymName(instance);
        if (!instSym) {
          instSym = StringAttr::get(
              context, il.mic.modNamespace.newName(instance.getName()));
          instance.setInnerSymAttr(hw::InnerSymAttr::get(instSym));
        }
        instOpHierPaths[InnerRefAttr::get(moduleName, instSym)].push_back(
            newSym);
        // TODO: Update any symbol renames which need to be used by the next
        // call of inlineInto.  This will then check each instance and rename
        // any symbols appropriately for that instance.
        symbolRenames.insert({origNLAName, newSym});
      }
    }
    auto instInnerSym = getInnerSymName(instance);
    auto parentActivePaths = activeHierpaths;
    setActiveHierPaths(moduleName, instInnerSym);
    // This must be done after the reTop, since it might introduce an innerSym.
    currentPath.emplace_back(moduleName, instInnerSym);

    InliningLevel childIL(il.mic, childModule);
    createDebugScope(childIL, instance, il.debugScope);

    // Create the wire mapping for results + ports.
    auto nestedPrefix = (prefix + instance.getName() + "_").str();
    mapPortsToWires(nestedPrefix, childIL, mapper, {});
    mapResultsToWires(mapper, childIL.wires, instance);

    // Inline the module, it can be marked as flatten and inline.
    if (toBeFlattened) {
      if (failed(flattenInto(nestedPrefix, childIL, mapper, {})))
        return failure();
    } else {
      if (failed(inlineInto(nestedPrefix, childIL, mapper, symbolRenames)))
        return failure();
    }
    currentPath.pop_back();
    activeHierpaths = parentActivePaths;
    return success();
  };

  return inliningWalk(il.mic.b, target.getBodyBlock(), mapper, visit);
}

LogicalResult Inliner::inlineInstances(FModuleOp module) {
  // Generate a namespace for this module so that we can safely inline
  // symbols.
  auto moduleName = module.getNameAttr();
  ModuleInliningContext mic(module);

  auto visit = [&](FInstanceLike instanceLike) {
    auto instance = dyn_cast<InstanceOp>(*instanceLike);
    if (!instance) {
      markUnknownFInstanceLikeModulesLive(instanceLike);
      return WalkResult::advance();
    }
    // If it's not a regular module we can't inline it. Mark it as live.
    auto *childModule = symbolTable.lookup(instance.getModuleName());
    auto target = dyn_cast<FModuleOp>(childModule);
    if (!target) {
      liveModules.insert(childModule);
      return WalkResult::advance();
    }

    // If we aren't inlining the target, add it to the work list.
    if (!shouldInline(target)) {
      if (liveModules.insert(target).second) {
        worklist.push_back(target);
      }
      return WalkResult::advance();
    }

    if (failed(checkInstanceParents(instance)))
      return WalkResult::interrupt();

    auto toBeFlattened = shouldFlatten(target);
    if (auto instSym = getInnerSymName(instance)) {
      auto innerRef = InnerRefAttr::get(moduleName, instSym);
      // Preorder update of any non-local annotations this instance participates
      // in.  This needs to happen _before_ visiting modules so that internal
      // non-local annotations can be deleted if they are now local.
      for (auto sym : instOpHierPaths[innerRef]) {
        auto *mnla = nlaMap[sym];
        // Skip if this NLA is rooted at target: it has been (or will be)
        // retop'd by the rootMap block below; calling inlineModule on the
        // root asserts.
        if (mnla->getNLA().root() == target.getNameAttr())
          continue;
        if (toBeFlattened)
          mnla->flattenModule(target);
        else
          mnla->inlineModule(target);
      }
    }

    // The InstanceOp `instance` might not have a symbol, if it does not
    // participate in any HierPathOp. But the reTop might add a symbol to it, if
    // a HierPathOp is added to this Op.
    DenseMap<Attribute, Attribute> symbolRenames;
    if (!rootMap[target.getNameAttr()].empty() && !toBeFlattened) {
      for (auto origSymAttr : rootMap[target.getNameAttr()]) {
        auto origSym = cast<StringAttr>(origSymAttr);
        auto *mnla = nlaMap[origSym];
        auto origNLAName = mnla->getNLA().getNameAttr();
        StringAttr newSym = mnla->reTop(module);
        if (newSym != origSym)
          nlaMap[newSym] = mnla;
        StringAttr instSym = getOrAddInnerSym(
            instance, [&](FModuleLike mod) -> hw::InnerSymbolNamespace & {
              return mic.modNamespace;
            });
        instOpHierPaths[InnerRefAttr::get(moduleName, instSym)].push_back(
            newSym);
        // TODO: Update any symbol renames which need to be used by the next
        // call of inlineInto.  This will then check each instance and rename
        // any symbols appropriately for that instance.
        symbolRenames.insert({origNLAName, newSym});
      }
    }
    auto instInnerSym = getInnerSymName(instance);
    auto parentActivePaths = activeHierpaths;
    setActiveHierPaths(moduleName, instInnerSym);
    // This must be done after the reTop, since it might introduce an innerSym.
    currentPath.emplace_back(moduleName, instInnerSym);
    // Create the wire mapping for results + ports. We RAUW the results instead
    // of mapping them.
    IRMapping mapper;
    mic.b.setInsertionPoint(instance);
    auto nestedPrefix = (instance.getName() + "_").str();

    InliningLevel childIL(mic, target);
    createDebugScope(childIL, instance);

    mapPortsToWires(nestedPrefix, childIL, mapper, {});
    for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i)
      instance.getResult(i).replaceAllUsesWith(childIL.wires[i]);

    // Inline the module, it can be marked as flatten and inline.
    if (toBeFlattened) {
      if (failed(flattenInto(nestedPrefix, childIL, mapper, {})))
        return WalkResult::interrupt();
    } else {
      // Recursively inline all the child modules under `parent`, that are
      // marked to be inlined.
      if (failed(inlineInto(nestedPrefix, childIL, mapper, symbolRenames)))
        return WalkResult::interrupt();
    }
    currentPath.pop_back();
    activeHierpaths = parentActivePaths;

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

void Inliner::identifyNLAsTargetingOnlyModules() {
  DenseSet<Operation *> nlaTargetedModules;

  // Identify candidate NLA's: those that end in a module.  (Iterate
  // sourceSymsInOrder so we visit each MutableNLA exactly once, even though
  // nlaMap may contain multiple aliases per source after reTop.  This is
  // called before any reTop has run, so this is moot at this point, but
  // robust regardless.)
  for (auto sym : sourceSymsInOrder) {
    auto *mnla = nlaMap[sym];
    auto nla = mnla->getNLA();
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

  // Mark NLA's that were not referenced as module-only.
  for (auto sym : sourceSymsInOrder) {
    auto *mnla = nlaMap[sym];
    auto nla = mnla->getNLA();
    if (nla.isModule() && !nonModOnlyNLAs.count(nla.getSymNameAttr()))
      mnla->markModuleOnly();
  }
}

Inliner::Inliner(CircuitOp circuit, SymbolTable &symbolTable)
    : circuit(circuit), context(circuit.getContext()),
      symbolTable(symbolTable) {}

LogicalResult Inliner::run() {
  CircuitNamespace circuitNamespace(circuit);

  // Gather all NLA's, build information about the instance ops used.
  // Reserve nlaStorage up-front so push_back below cannot reallocate it
  // (pointers stored in nlaMap must remain stable throughout the pass).
  auto hierPaths = circuit.getBodyBlock()->getOps<hw::HierPathOp>();
  size_t numSources = std::distance(hierPaths.begin(), hierPaths.end());
  nlaStorage.reserve(numSources);
  for (auto nla : hierPaths) {
    nlaStorage.emplace_back(nla, &circuitNamespace);
    auto *p = &nlaStorage.back();
    auto symName = nla.getSymNameAttr();
    // Default context's outputSym IS the source sym, so a single nlaMap
    // entry suffices initially.  reTop may register additional aliases.
    nlaMap[symName] = p;
    sourceSymsInOrder.push_back(symName);
    rootMap[p->getNLA().root()].push_back(symName);
    for (auto path : nla.getNamepath())
      if (auto ref = dyn_cast<InnerRefAttr>(path))
        instOpHierPaths[ref].push_back(symName);
  }
  // Mark 'module-only' the NLA's that only target modules.
  // These may be deleted when their module is inlined/flattened.
  identifyNLAsTargetingOnlyModules();

  // Mark the top module as live, so it doesn't get deleted.
  for (auto &op : circuit.getOps()) {
    // Mark public/non-discardable modules as live and add them to the worklist.
    if (auto module = dyn_cast<FModuleLike>(op)) {
      if (module.canDiscardOnUseEmpty())
        continue;
      liveModules.insert(module);
      if (isa<FModuleOp>(module))
        worklist.push_back(cast<FModuleOp>(module));
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
          if (liveModules.insert(moduleLike).second)
            if (auto module = dyn_cast<FModuleOp>(*moduleLike))
              worklist.push_back(module);
    }
  }

  // If the module is marked for flattening, flatten it. Otherwise, inline
  // every instance marked to be inlined.
  while (!worklist.empty()) {
    auto moduleOp = worklist.pop_back_val();
    if (shouldFlatten(moduleOp)) {
      if (failed(flattenInstances(moduleOp)))
        return failure();
      // Delete the flatten annotation, the transform was performed.
      // Even if visited again in our walk (for inlining),
      // we've just flattened it and so the annotation is no longer needed.
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

  // Delete all unreferenced modules.  Mark any NLAs that originate from
  // dead modules as also dead -- but only if the NLA hasn't been retop'd
  // (retop'd NLAs have moved to a new live root and produce per-context
  // outputs; their source root module being inlined away is expected).
  for (auto mod : llvm::make_early_inc_range(
           circuit.getBodyBlock()->getOps<FModuleLike>())) {
    if (liveModules.count(mod))
      continue;
    for (auto nla : rootMap[mod.getModuleNameAttr()]) {
      auto *mnla = nlaMap[nla];
      if (!mnla->isRetopped())
        mnla->markDead();
    }
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
    assert(!shouldFlatten(mod) && "flatten annotation found on live module");
  }

  LLVM_DEBUG({
    llvm::dbgs() << "NLA modifications:\n";
    for (auto nla : circuit.getBodyBlock()->getOps<hw::HierPathOp>())
      nlaMap[nla.getNameAttr()]->dump();
  });

  // Writeback all NLAs to MLIR.  Each MutableNLA emits one new
  // hw.hierpath per context (or leaves the source op unchanged if nothing
  // changed).  Iterate in IR source order for a deterministic layout.
  for (auto sym : sourceSymsInOrder)
    nlaMap.find(sym)->second->applyUpdates();

  // Update non-local annotations at non-root modules to reflect the
  // post-inlining NLA state.
  for (auto fmodule : circuit.getBodyBlock()->getOps<FModuleLike>()) {
    SmallVector<Attribute> newAnnotations;
    auto processNLAs = [&](Annotation anno) -> bool {
      if (auto sym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
        auto it = nlaMap.find(sym.getAttr());
        if (it == nlaMap.end())
          return false;
        auto *mnla = it->second;

        // NLA was killed; drop the annotation.
        if (mnla->isDead())
          return true;

        // NLA became local after inlining; strip the nonlocal marker.
        if (mnla->isLocal()) {
          anno.removeMember("circt.nonlocal");
          newAnnotations.push_back(anno.getAttr());
          return true;
        }

        // Each instantiation context needs its own annotation copy.
        // Root modules are handled during instance renaming; skip them here.
        auto outputSyms = mnla->getOutputSyms();
        if (outputSyms.size() <= 1 || mnla->hasRoot(fmodule))
          return false;

        // The annotation already carries the first output sym; emit one
        // additional copy per extra context.
        for (auto outSym : ArrayRef(outputSyms).drop_front()) {
          NamedAttrList newAnnotation;
          for (auto pair : anno.getDict()) {
            if (pair.getName().getValue() != "circt.nonlocal") {
              newAnnotation.push_back(pair);
              continue;
            }
            newAnnotation.push_back(
                {pair.getName(), FlatSymbolRefAttr::get(outSym)});
          }
          newAnnotations.push_back(DictionaryAttr::get(context, newAnnotation));
        }
      }
      return false;
    };
    fmodule.walk([&](Operation *op) {
      AnnotationSet annotations(op);
      // Early exit to avoid adding an empty annotations attribute to operations
      // which did not previously have annotations.
      if (annotations.empty())
        return;

      // Update annotations on the op.
      newAnnotations.clear();
      annotations.removeAnnotations(processNLAs);
      annotations.addAnnotations(newAnnotations);
      annotations.applyToOperation(op);
    });

    // Update annotations on the ports.
    SmallVector<Attribute> newPortAnnotations;
    for (auto port : fmodule.getPorts()) {
      newAnnotations.clear();
      port.annotations.removeAnnotations(processNLAs);
      port.annotations.addAnnotations(newAnnotations);
      newPortAnnotations.push_back(
          ArrayAttr::get(context, port.annotations.getArray()));
    }
    fmodule->setAttr("portAnnotations",
                     ArrayAttr::get(context, newPortAnnotations));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class InlinerPass : public circt::firrtl::impl::InlinerBase<InlinerPass> {
  void runOnOperation() override {
    CIRCT_DEBUG_SCOPED_PASS_LOGGER(this);
    Inliner inliner(getOperation(), getAnalysis<SymbolTable>());
    if (failed(inliner.run()))
      signalPassFailure();
  }
};
} // namespace
