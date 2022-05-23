//===- GrandCentralSignalMappings.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GrandCentralSignalMappings pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "gct"

namespace json = llvm::json;

using namespace circt;
using namespace firrtl;

//===- Terminology and useful information ---------------------------------===//
//
// For SignalDriver annotations, "local" refers to the subcircuit that
// drives/probes values in the main "remote" circuit.
//
// Driven values ("sinks"), are set via 'force XMR = input_port;' statements.
// Probes ("sources"), are read via 'assign output_port = XMR;' statements.
//
// The end goal is a mappings module that is generated and instantiated in the
// subcircuit which interacts with targeted values in the main circuit.
//
//===- Flow ---------------------------------------------------------------===//
//
// Signal driver annotations are needed when processing both circuits:
//
// First the original annotations are used with the main circuit, in order to
// emit extra wires around forced inputs and to emit updated annotations
// pointing to the values across naming or hierarchy changes performed.
//
// Second these updated annotations are used when processing the subcircuit
// which is when the mappings module is created and instantiated.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Per-Module Signal Mappings
//===----------------------------------------------------------------------===//

namespace {
enum class MappingDirection {
  /// The local signal forces the remote signal.
  DriveRemote, // Sink
  /// The local signal is driven to the value of the remote signal through a
  /// connect.
  ProbeRemote, // Source
};

/// The information necessary to connect a local signal in a module to a remote
/// value (in a different module and/or circuit).
struct SignalMapping {
  /// Whether we force or read from the target.
  MappingDirection dir;
  /// The reference target of the thing we are forcing or probing.
  StringAttr remoteTarget;
  /// The type of the signal being mapped.
  FIRRTLType type;
  /// The block argument or result that interacts with the remote target, either
  /// by forcing it or by reading from it through a connect.
  Value localValue;
  /// The name of the local value, for reuse in the generated signal mappings
  /// module.
  StringAttr localName;
  /// Which "side" we're on
  bool isLocal;
  /// NLA, if present
  FlatSymbolRefAttr nlaSym;
};

/// A helper structure that collects the data necessary to generate the signal
/// mappings module for an existing `FModuleOp` in the IR.
struct ModuleSignalMappings {
  ModuleSignalMappings(FModuleOp module, StringRef markDut, StringRef prefix)
      : module(module), markDut(markDut), prefix(prefix) {}
  void run();
  void addTarget(Value value, Annotation anno);
  FModuleOp emitMappingsModule();
  void instantiateMappingsModule(FModuleOp mappingsModule);

  FModuleOp module;
  bool allAnalysesPreserved = true;
  SmallVector<SignalMapping> mappings;
  SmallString<64> mappingsModuleName;

  StringRef markDut;
  StringRef prefix;
};
} // namespace

// Allow `SignalMapping` to be printed.
template <typename T>
static T &operator<<(T &os, const SignalMapping &mapping) {
  os << "SignalMapping { remote"
     << (mapping.dir == MappingDirection::DriveRemote ? "Sink" : "Source")
     << ": " << mapping.remoteTarget << ", "
     << "localTarget: " << mapping.localName << " }";
  return os;
}

/// Analyze the `module` of this `ModuleSignalMappings` and generate the
/// corresponding auxiliary `FModuleOp` with the necessary cross-module
/// references and `ForceOp`s to probe and drive remote signals. This is
/// dictated by the presence of `SignalDriverAnnotation` on the module and
/// individual operations inside it.
/// If the module is the "remote" (main) circuit, gather those mappings
/// for use handling forced input ports and creating updated mappings.
void ModuleSignalMappings::run() {
  // Check whether this module has any `SignalDriverAnnotation`s. These indicate
  // whether the module contains any operations with such annotations and
  // requires processing.
  if (!AnnotationSet::removeAnnotations(module, signalDriverAnnoClass)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping `" << module.getName()
                            << "` (has no annotations)\n");
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Running on module `" << module.getName()
                          << "`\n");

  // Gather the signal driver annotations on the ports of this module.
  LLVM_DEBUG(llvm::dbgs() << "- Gather port annotations\n");
  AnnotationSet::removePortAnnotations(
      module, [&](unsigned i, Annotation anno) {
        if (!anno.isClass(signalDriverAnnoClass))
          return false;
        addTarget(module.getArgument(i), anno);
        return true;
      });

  // Gather the signal driver annotations of the operations within this module.
  LLVM_DEBUG(llvm::dbgs() << "- Gather operation annotations\n");
  module.walk([&](Operation *op) {
    AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
      if (!anno.isClass(signalDriverAnnoClass))
        return false;
      for (auto result : op->getResults())
        addTarget(result, anno);
      return true;
    });
  });

  auto localMappings =
      llvm::make_filter_range(mappings, [](auto &m) { return m.isLocal; });

  // Remove connections to sources in the subcircuit.  This is done to cleanup
  // invalidations that occur from the Chisel API of Grand Central.
  // Only the remote value "source" should be connected to local source values.
  for (auto mapping : localMappings)
    if (mapping.dir == MappingDirection::ProbeRemote) {
      for (auto &use : llvm::make_early_inc_range(mapping.localValue.getUses()))
        if (auto connect = dyn_cast<FConnectLike>(use.getOwner()))
          if (connect.dest() == mapping.localValue) {
            connect.erase();
            allAnalysesPreserved = false;
          };
    }

  // If there aren't any local-side (subcircuit-side) mappings, we're done.
  if (localMappings.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping `" << module.getName()
                            << "` (has no non-zero sources or sinks)\n");
    return;
  }

  // Changes are coming, mark analyses not preserved
  allAnalysesPreserved = false;

  if (!llvm::all_of(mappings, [](auto &m) { return m.isLocal; })) {
    emitWarning(module->getLoc(),
                "both remote and local SignalDriverAnnotation mappings found");
  }

  // Pick a name for the module that implements the signal mappings.
  CircuitNamespace circuitNamespace(module->getParentOfType<CircuitOp>());
  mappingsModuleName =
      circuitNamespace.newName(Twine(module.getName()) + "_signal_mappings");

  // Generate the mappings module.
  auto mappingsModule = emitMappingsModule();

  // Instantiate the mappings module.
  instantiateMappingsModule(mappingsModule);
}

/// Mark a `value` inside the `module` as being the target of the
/// `SignalDriverAnnotation` `anno`. This generates the necessary
/// `SignalMapping` information and adds an entry to the `mappings` array, to be
/// later consumed when the mappings module is constructed.
void ModuleSignalMappings::addTarget(Value value, Annotation anno) {
  // Ignore the target if it is zero width.
  if (!value.getType().cast<FIRRTLType>().getBitWidthOrSentinel())
    return;

  SignalMapping mapping;
  mapping.dir = anno.getMember<StringAttr>("dir").getValue() == "source"
                    ? MappingDirection::ProbeRemote
                    : MappingDirection::DriveRemote;
  mapping.remoteTarget = anno.getMember<StringAttr>("peer");
  mapping.localValue = value;
  mapping.type = value.getType().cast<FIRRTLType>();
  mapping.isLocal = anno.getMember<StringAttr>("side").getValue() == "local";
  mapping.nlaSym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");

  // Guess a name for the local value. This is only for readability's sake,
  // giving the pass a hint for picking the names of the generated module ports.
  if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    mapping.localName = module.getPortNameAttr(blockArg.getArgNumber());
  } else if (auto *op = value.getDefiningOp()) {
    mapping.localName = op->getAttrOfType<StringAttr>("name");
  }

  LLVM_DEBUG(llvm::dbgs() << "  - " << mapping << "\n");
  mappings.push_back(std::move(mapping));
}

/// Create a separate mappings module that contains cross-module references and
/// `ForceOp`s for each entry in the `mappings` array.
FModuleOp ModuleSignalMappings::emitMappingsModule() {
  LLVM_DEBUG(llvm::dbgs() << "- Generating `" << mappingsModuleName << "`\n");

  // Determine what ports this module will have, given the signal mappings we
  // are supposed to emit.
  SmallVector<PortInfo> ports;
  auto localMappings =
      llvm::make_filter_range(mappings, [](auto &m) { return m.isLocal; });
  for (auto &mapping : localMappings) {
    ports.push_back(PortInfo{mapping.localName,
                             mapping.type,
                             mapping.dir == MappingDirection::DriveRemote
                                 ? Direction::In
                                 : Direction::Out,
                             {},
                             module.getLoc()});
    LLVM_DEBUG(llvm::dbgs() << "  - Adding port " << mapping.localName << "\n");
  }

  // Create the actual module.
  ImplicitLocOpBuilder builder(module.getLoc(), module);
  auto mappingsModule = builder.create<FModuleOp>(
      StringAttr::get(module.getContext(), mappingsModuleName), ports);

  // Generate the connect and force statements inside the module.
  builder.setInsertionPointToStart(mappingsModule.getBody());
  unsigned portIdx = 0;
  for (auto &mapping : localMappings) {
    // TODO: Actually generate a proper XMR here. For now just do some textual
    // replacements. Generating a real IR node (like a proper XMR op) would be
    // much better, but the modules that `EmitSignalMappings` interacts with
    // generally live in a separate circuit. Multiple circuits are not fully
    // supported at the moment.
    SmallString<32> remoteXmrName;
    auto [circuitName, pathName] = mapping.remoteTarget.getValue().split('|');
    // llvm::errs() << "Rewriting " << circuitName << " " << pathName <<
    // "\nWith: " << markDut << " " << Prefix << "\n";
    bool seenRoot = false;
    auto [modulePath, varPath] = pathName.split('>');
    if (markDut.empty()) {
      // If no DUT, top-level is first module in modulePath
      // ("~Top|Foo/b:B" -> "Foo.b")
      remoteXmrName += modulePath.split('/').first;
      seenRoot = true;
    }
    do {
      auto [item, tail] = modulePath.split(':');
      modulePath = tail;
      auto [modName, instName] = item.split('/');
      if (!markDut.empty() && markDut == modName) {
        remoteXmrName += prefix;
        remoteXmrName += markDut;
        seenRoot = true;
      }
      if (tail.empty())
        break;
      if (!markDut.empty() && !seenRoot)
        continue;
      if (!instName.empty()) {
        remoteXmrName += '.';
        remoteXmrName += instName;
      }
    } while (true);
    remoteXmrName.push_back('.');
    for (auto c : varPath) {
      if (c == '[' || c == '.')
        remoteXmrName.push_back('_');
      else if (c != ']')
        remoteXmrName.push_back(c);
    }
    // llvm::errs() << "XMR: " << remoteXmrName << "\n\n";
    if (mapping.dir == MappingDirection::DriveRemote) {
      auto xmr = builder.create<VerbatimWireOp>(mapping.type, remoteXmrName);
      builder.create<ForceOp>(xmr, mappingsModule.getArgument(portIdx++));
    } else {
      auto xmr = builder.create<VerbatimWireOp>(mapping.type, remoteXmrName);
      builder.create<ConnectOp>(mappingsModule.getArgument(portIdx++), xmr);
    }
  }
  return mappingsModule;
}

/// Instantiate the generated mappings module inside the `module` we are working
/// on, and generated the necessary connections to local signals.
void ModuleSignalMappings::instantiateMappingsModule(FModuleOp mappingsModule) {
  LLVM_DEBUG(llvm::dbgs() << "- Instantiating `" << mappingsModuleName
                          << "`\n");
  // Create the actual module.
  auto builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());
  auto inst = builder.create<InstanceOp>(mappingsModule, "signal_mappings");

  // Generate the connections to and from the instance.
  unsigned portIdx = 0;
  auto localMappings =
      llvm::make_filter_range(mappings, [](auto &m) { return m.isLocal; });
  for (auto &mapping : localMappings) {
    Value dst = inst.getResult(portIdx++);
    Value src = mapping.localValue;
    if (mapping.dir == MappingDirection::ProbeRemote)
      std::swap(src, dst);
    builder.create<ConnectOp>(dst, src);
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

/// External modules, for emitting when processing subcircuit
struct ExtModules {
  /// External modules put in the vsrcs field of the JSON.
  SmallVector<FExtModuleOp> vsrc;
  /// External modules put in the "load_jsons" field of the JSON.
  SmallVector<FExtModuleOp> json;
};

/// Information gathered about mappings, used for identifying forced input ports
/// and generating updated mappings for the remote side (main circuit).
struct ModuleMappingResult {
  /// Were changes made that invalidate analyses?
  bool allAnalysesPreserved;
  /// Module scanned
  FModuleOp module;
  /// Signal mappings whose remote targets are in this circuit
  SmallVector<SignalMapping> remoteMappings;
};

class GrandCentralSignalMappingsPass
    : public GrandCentralSignalMappingsBase<GrandCentralSignalMappingsPass> {

  /// Generate operation to output list of external modules
  /// Used when processing the sub- (local) circuit.
  void emitSubCircuitJSON(CircuitOp circuit, StringAttr circuitPackage,
                          StringRef outputFilename,
                          const ExtModules &extmodules);

  /// Fixup forced input ports and emit updated mappings
  /// Used when processing the main (remote) circuit.
  FailureOr<bool>
  emitUpdatedMappings(CircuitOp circuit, StringAttr circuitPackage,
                      MutableArrayRef<ModuleMappingResult> result);

  void runOnOperation() override;

public:
  std::string outputFilename;
  std::string markDut;
  std::string prefix;
};

void GrandCentralSignalMappingsPass::runOnOperation() {
  CircuitOp circuit = getOperation();

  StringAttr circuitPackage;
  BoolAttr isSubCircuitAttr;
  bool circuitHasAnno =
      AnnotationSet::removeAnnotations(circuit, [&](Annotation anno) {
        if (!anno.isClass(signalDriverAnnoClass))
          return false;

        isSubCircuitAttr = anno.getMember<BoolAttr>("isSubCircuit");
        circuitPackage = anno.getMember<StringAttr>("circuitPackage");
        return true;
      });

  if (!circuitHasAnno) {
    // Nothing to do here
    markAllAnalysesPreserved();
    return;
  }

  if (!isSubCircuitAttr) {
    emitError(
        circuit->getLoc(),
        "has invalid SignalDriverAnnotation (missing 'isSubCircuit' field')");
    return signalPassFailure();
  }
  bool isSubCircuit = isSubCircuitAttr.getValue();

  if (isSubCircuit && !circuitPackage) {
    emitError(circuit->getLoc())
        << "has invalid SignalDriverAnnotation (subcircuit JSON emission is "
           "enabled, but no circuitPackage was provided)";
    return signalPassFailure();
  }

  SmallVector<FModuleOp> modules;
  ExtModules extmodules;

  for (auto op : circuit.body().getOps<FModuleLike>()) {
    if (auto *extModule = dyn_cast<FExtModuleOp>(&op)) {
      AnnotationSet annotations(*extModule);
      if (annotations.hasAnnotation("firrtl.transforms.BlackBoxInlineAnno")) {
        extmodules.vsrc.push_back(*extModule);
        continue;
      }
      extmodules.json.push_back(*extModule);
    } else if (auto *module = dyn_cast<FModuleOp>(&op)) {
      modules.push_back(*module);
    }
  }

  // Pre-allocate result vector
  SmallVector<ModuleMappingResult> results;
  results.resize(modules.size());

  mlir::parallelForEachN(
      circuit.getContext(), 0, modules.size(),
      [this, &modules, &results](size_t index) {
        ModuleSignalMappings mapper(modules[index], markDut, prefix);
        mapper.run();
        results[index] = {
            mapper.allAnalysesPreserved, modules[index],
            SmallVector<SignalMapping>{llvm::make_filter_range(
                mapper.mappings, [](auto &m) { return !m.isLocal; })}};
      });
  bool allAnalysesPreserved =
      llvm::all_of(results, [](auto &r) { return r.allAnalysesPreserved; });

  // If this is a subcircuit, then emit JSON information necessary to drive
  // SiFive tools.
  // Otherwise handle any forced input ports and emit updated mappings.
  if (isSubCircuit) {
    emitSubCircuitJSON(circuit, circuitPackage, outputFilename, extmodules);
  } else {
    auto result = emitUpdatedMappings(circuit, circuitPackage, results);
    if (failed(result))
      return signalPassFailure();
    allAnalysesPreserved &= *result;
  }

  if (allAnalysesPreserved)
    markAllAnalysesPreserved();
}

void GrandCentralSignalMappingsPass::emitSubCircuitJSON(
    CircuitOp circuit, StringAttr circuitPackage, StringRef outputFilename,
    const ExtModules &extmodules) {
  std::string jsonString;
  llvm::raw_string_ostream jsonStream(jsonString);
  json::OStream j(jsonStream, 2);
  j.object([&] {
    j.attributeObject("vendor", [&]() {
      j.attributeObject("vcs", [&]() {
        j.attributeArray("vsrcs", [&]() {
          for (FModuleOp module : circuit.body().getOps<FModuleOp>()) {
            SmallVector<char> file(outputFilename.begin(),
                                   outputFilename.end());
            llvm::sys::fs::make_absolute(file);
            llvm::sys::path::append(file, Twine(module.moduleName()) + ".sv");
            j.value(file);
          }
          for (FExtModuleOp ext : extmodules.vsrc) {
            SmallVector<char> file(outputFilename.begin(),
                                   outputFilename.end());
            llvm::sys::fs::make_absolute(file);
            llvm::sys::path::append(file, Twine(ext.moduleName()) + ".sv");
            j.value(file);
          }
        });
      });
      j.attributeObject("verilator", [&]() {
        j.attributeArray("error", [&]() {
          j.value("force statement is not supported in verilator");
        });
      });
    });
    j.attributeArray("remove_vsrcs", []() {});
    j.attributeArray("vsrcs", []() {});
    j.attributeArray("load_jsons", [&]() {
      for (FExtModuleOp extModule : extmodules.json)
        j.value((Twine(extModule.moduleName()) + ".json").str());
    });
  });
  auto b = OpBuilder::atBlockEnd(circuit.getBody());
  auto jsonOp = b.create<sv::VerbatimOp>(b.getUnknownLoc(), jsonString);
  jsonOp->setAttr(
      "output_file",
      hw::OutputFileAttr::getFromFilename(
          b.getContext(), Twine(circuitPackage) + ".subcircuit.json", true));
}

FailureOr<bool> GrandCentralSignalMappingsPass::emitUpdatedMappings(
    CircuitOp circuit, StringAttr circuitPackage,
    MutableArrayRef<ModuleMappingResult> results) {
  bool allAnalysesPreserved = true;

  // 1) Sanity checking
  // 2) Fixup ports that are driven from subcircuit
  // 3) Generate new remote references
  //    For now, just drive/probe using `module.name` XMR's,
  //    it's much simpler than emitting hierarchical paths
  //    from the top module/DUT, and works for now.
  // 4) Emit updated mappings as SignalDriverAnnotations
  //    for consumption when processing the subcircuit
  //    (as replacements for the original annotations)

  // Helper to instantiate module namespaces as-needed
  DenseMap<FModuleOp, ModuleNamespace> moduleNamespaces;
  auto getModuleNamespace =
      [&moduleNamespaces](FModuleOp module) -> ModuleNamespace & {
    return moduleNamespaces.try_emplace(module, module).first->second;
  };

  // (1) Scan mappings for unexpected or unsupported values
  for (auto const &[_, mod, mappings] : results) {
    for (auto &mapping : mappings) {
      // This isn't handled yet, error out instead of doing wrong thing
      if (mapping.nlaSym) {
        emitError(mapping.localValue.getLoc())
            << "non-local value targeted by SignalDriverAnnotation, "
               "unsupported (NLA found)";
        return failure();
      }
      // No local-side mappings on main circuit side
      if (mapping.isLocal) {
        emitError(mapping.localValue.getLoc())
            << "local-side SignalDriverAnnotation found in main circuit, "
               "mixing not supported";
        return failure();
      }
    }
  }

  // (2) Find input ports that are sinks, insert wires at instantiation points
  auto *instanceGraph = &getAnalysis<InstanceGraph>();
  DenseSet<unsigned> forcedInputPorts;
  for (auto &[_, mod, mappings] : results) {
    // Walk mappings looking for module ports that are being driven,
    // and gather their indices for fixing up.
    forcedInputPorts.clear();
    for (auto &mapping : mappings) {
      if (mapping.dir == MappingDirection::DriveRemote /* Sink */) {
        if (auto blockArg = mapping.localValue.dyn_cast<BlockArgument>()) {
          auto portIdx = blockArg.getArgNumber();
          if (mod.getPortDirection(portIdx) == Direction::In) {
            if (!forcedInputPorts.insert(portIdx).second) {
              emitError(blockArg.getLoc())
                  << "module port driven more than once, unsupported "
                     "SignalDriverAnnotation";
              return failure();
            }
          }
        }
      }
    }

    // Find all instantiations of this module, and replace uses of each driven
    // port with a wire that's connected to a wire that is connected to the
    // port. This is done to cause an 'assign' to be created, disconnecting
    // the forced input port's net from its uses.
    if (forcedInputPorts.empty())
      continue;

    unsigned useCount = 0;
    for (auto *use : instanceGraph->lookup(mod)->uses()) {
      // Ensure just the one use for now, this matters if we want
      // to be able to emit a unique path from the top module.
      if (++useCount != 1) {
        emitError(mod.getLoc())
            << "module with input port driven instantiated more than once, "
               "unsupported SignalDriverAnnotation";

        return failure();
      }
      allAnalysesPreserved = false;

      auto inst = use->getInstance();
      OpBuilder builder(inst.getContext());
      builder.setInsertionPointAfter(inst);
      auto parentModule = inst->getParentOfType<FModuleOp>();
      ModuleNamespace &moduleNamespace = getModuleNamespace(parentModule);

      for (auto portIdx : forcedInputPorts) {
        auto port = inst->getResult(portIdx);

        // Create chain like:
        // port_result <= foo_dataIn_x_buffer
        // foo_dataIn_x_buffer <= foo_dataIn_x
        auto replacementWireName =
            builder.getStringAttr(moduleNamespace.newName(
                mod.moduleName() + "_" + mod.getPortName(portIdx)));
        auto bufferWireName = builder.getStringAttr(moduleNamespace.newName(
            replacementWireName.getValue() + "_buffer"));
        auto bufferWire = builder.create<WireOp>(
            builder.getUnknownLoc(), port.getType(), bufferWireName,
            builder.getArrayAttr({}), bufferWireName);
        auto replacementWire = builder.create<WireOp>(
            builder.getUnknownLoc(), port.getType(), replacementWireName,
            builder.getArrayAttr({}), replacementWireName);
        port.replaceAllUsesWith(replacementWire);
        builder.create<StrictConnectOp>(builder.getUnknownLoc(), port,
                                        bufferWire);
        builder.create<StrictConnectOp>(builder.getUnknownLoc(), bufferWire,
                                        replacementWire);
      }
    }
  }

  // (3) Generate updated remote references
  // Helpers to emit basic 'module.name' XMR's for the mappings.
  // Use inner_sym for non-ports
  SmallVector<Attribute> symbols;
  auto getOrAddInnerSym = [&](Operation *op) -> StringAttr {
    auto attr = op->getAttrOfType<StringAttr>("inner_sym");
    if (attr)
      return attr;
    StringRef name = "sym";
    if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
      name = nameAttr.getValue();
    auto module = op->getParentOfType<FModuleOp>();
    name = getModuleNamespace(module).newName(name);
    attr = StringAttr::get(op->getContext(), name);
    op->setAttr("inner_sym", attr);
    return attr;
  };
  auto mkRef = [&](FModuleOp module,
                   const SignalMapping &mapping) -> std::string {
    if (mapping.localValue.isa<BlockArgument>()) {
      return llvm::formatv("~{0}|{1}>{2}", circuit.name(), module.getName(),
                           mapping.localName);
    }
    auto idx = symbols.size();
    symbols.push_back(hw::InnerRefAttr::get(
        SymbolTable::getSymbolName(module),
        getOrAddInnerSym(mapping.localValue.getDefiningOp())));
    return llvm::formatv("~{0}|{1}>{2}", circuit.name(), module.getName(),
                         "{{" + Twine(idx) + "}}");
  };

  // Generate and sort new mappings for (better) output stability
  SmallVector<std::pair<std::string, StringRef>, 32> sinks, sources;
  auto addUpdatedMappings = [&](MappingDirection filterDir, auto &vec) {
    for (auto const &[_, mod, mappings] : results)
      for (auto &mapping : mappings)
        if (mapping.dir == filterDir)
          vec.emplace_back(mkRef(mod, mapping),
                           mapping.remoteTarget.getValue());
  };
  addUpdatedMappings(MappingDirection::DriveRemote, sinks);
  addUpdatedMappings(MappingDirection::ProbeRemote, sources);
  // Sort on local targets only, remote may contain substitution placeholders
  llvm::sort(sinks, llvm::less_second());
  llvm::sort(sources, llvm::less_second());

  // (4) Emit updated mappings for consumption by subcircuit
  std::string jsonString;
  llvm::raw_string_ostream jsonStream(jsonString);
  json::OStream j(jsonStream, 2);

  j.array([&] {
    j.object([&] {
      j.attribute("class", signalDriverAnnoClass);
      j.attributeArray("sinkTargets", [&]() {
        for (auto &[remote, local] : sinks) {
          j.objectBegin();
          j.attribute("_1", remote);
          j.attribute("_2", local);
          j.objectEnd();
        };
      });
      j.attributeArray("sourceTargets", [&]() {
        for (auto &[remote, local] : sources) {
          j.objectBegin();
          j.attribute("_1", remote);
          j.attribute("_2", local);
          j.objectEnd();
        };
      });
      // Emit these but don't attempt to plumb through their original values
      j.attribute("circuit", "circuit empty :\n  module empty :\n\n    skip\n");
      j.attributeArray("annotations", [&]() {});
      if (circuitPackage)
        j.attribute("circuitPackage", circuitPackage.getValue());
    });
  });
  auto b = OpBuilder::atBlockEnd(circuit.getBody());
  auto jsonOp = b.create<sv::VerbatimOp>(b.getUnknownLoc(), jsonString,
                                         ValueRange{}, b.getArrayAttr(symbols));
  // TODO: plumb option to specify path/name?
  constexpr char jsonOutFile[] = "sigdrive.json";
  jsonOp->setAttr("output_file", hw::OutputFileAttr::getFromFilename(
                                     b.getContext(), jsonOutFile, true));

  return allAnalysesPreserved;
}

std::unique_ptr<mlir::Pass> circt::firrtl::createGrandCentralSignalMappingsPass(
    StringRef outputFilename, StringRef markDut, StringRef prefix) {
  auto pass = std::make_unique<GrandCentralSignalMappingsPass>();
  if (!outputFilename.empty())
    pass->outputFilename = outputFilename;
  if (!markDut.empty())
    pass->markDut = markDut;
  if (!prefix.empty())
    pass->prefix = prefix;
  return pass;
}
