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
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "gct"

namespace json = llvm::json;

using namespace circt;
using namespace firrtl;

static constexpr const char *signalDriverAnnoClass =
    "sifive.enterprise.grandcentral.SignalDriverAnnotation";

//===----------------------------------------------------------------------===//
// Per-Module Signal Mappings
//===----------------------------------------------------------------------===//

namespace {
enum class MappingDirection {
  /// The local signal forces the remote signal.
  DriveRemote,
  /// The local signal is driven to the value of the remote signal through a
  /// connect.
  ProbeRemote,
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
  bool allAnalysesPreserved = false;
  SmallVector<SignalMapping> mappings;
  SmallString<64> mappingsModuleName;

  StringRef markDut;
  StringRef prefix;
  DenseSet<unsigned> forcedInputPorts;
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
void ModuleSignalMappings::run() {
  // Check whether this module has any `SignalDriverAnnotation`s. These indicate
  // whether the module contains any operations with such annotations and
  // requires processing.
  // AnnotationSet origAnnos(module);
  // llvm::errs() << "annos for: " << module.getName() << "\n";
  // for (auto a: origAnnos) {
  //   a.dump();
  // };
  if (!AnnotationSet::removeAnnotations(module, signalDriverAnnoClass)) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping `" << module.getName()
                            << "` (has no annotations)\n");
    allAnalysesPreserved = true;
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

  // Remove connections to sources.  This is done to cleanup invalidations that
  // occur from the Chisel API of Grand Central.
  for (auto mapping : mappings)
    if (mapping.dir == MappingDirection::ProbeRemote) {
      for (auto &use : llvm::make_early_inc_range(mapping.localValue.getUses()))
        if (auto connect = dyn_cast<FConnectLike>(use.getOwner()))
          if (connect.dest() == mapping.localValue)
            connect.erase();
    }

//  // JSON
//  for (auto mapping: mappings) {
//    std::string jsonString;
//    llvm::raw_string_ostream jsonStream(jsonString);
//    json::OStream j(jsonStream, /* indentSize */ 2);
//
//    auto b = OpBuilder::atBlockEnd(circuit.getBody());
//    auto jsonOp = b.create<sv::VerbatimOp>(b.getUnknownLoc(), jsonString);
//    jsonOp->setAttr(
//        "output_file",
//        hw::OutputFileAttr::getFromFilename(
//          b.getContext(), Twine(circuitPackage) + ".sigdrive.json", true));
//
//  }
//
  // If this module either
  if (mappings.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping `" << module.getName()
                            << "` (has no non-zero sources or sinks)\n");
    allAnalysesPreserved = true;
    return;
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

  // Only continue to emit signal driving code for the "local" side of these
  // annotations, which sits in the sub-circuit and interacts with the main
  // circuit on the "remote" side.  If we are in the "remote" side of the
  // annotation, which sits in the main circuit, then record if we ever see any
  // forces of module inputs.  These require special fixups due to the fact that
  // SV force will force the entire net connected to the port as well.
  if (anno.getMember<StringAttr>("side").getValue() != "local") {
    if (mapping.dir != MappingDirection::DriveRemote)
      return;
    auto blockArg = value.dyn_cast<BlockArgument>();
    if (!blockArg)
      return;
    auto portIdx = blockArg.getArgNumber();
    if (module.getPortDirection(portIdx) == Direction::Out)
      return;
    forcedInputPorts.insert(portIdx);
    return;
  }

  // Guess a name for the local value. This is only for readability's sake,
  // giving the pass a hint for picking the names of the generated module ports.
  if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    mapping.localName = module.getPortNameAttr(blockArg.getArgNumber());
  } else if (auto op = value.getDefiningOp()) {
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
  for (auto &mapping : mappings) {
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
  for (auto &mapping : mappings) {
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
    if (markDut.empty()) {
      remoteXmrName += circuitName.drop_front();
      seenRoot = true;
    }
    auto [modulePath, varPath] = pathName.split('>');
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
  for (auto &mapping : mappings) {
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

class GrandCentralSignalMappingsPass
    : public GrandCentralSignalMappingsBase<GrandCentralSignalMappingsPass> {
  void runOnOperation() override;

public:
  std::string outputFilename;
  std::string markDut;
  std::string prefix;
};

void GrandCentralSignalMappingsPass::runOnOperation() {
  CircuitOp circuit = getOperation();

  bool emitJSON = false;
  StringAttr circuitPackage;
  AnnotationSet::removeAnnotations(circuit, [&](Annotation anno) {
    if (!anno.isClass("sifive.enterprise.grandcentral.SignalDriverAnnotation"))
      return false;

    emitJSON = anno.getDict().contains("emitJSON");
    circuitPackage = anno.getMember<StringAttr>("circuitPackage");
    return true;
  });

  if (emitJSON && !circuitPackage) {
    emitError(circuit->getLoc())
        << "has invalid SignalDriverAnnotation (JSON emission is enabled, but "
           "no circuitPacakge was provided";
    return signalPassFailure();
  }

  typedef struct {
    bool allAnalysesPreserved;
    DenseMap<FModuleOp, DenseSet<unsigned>> forcedInputPorts;
  } Result;

  // Save gathered mappings
  // TODO: Thread-safe :(
  // DenseMap of SmallVector :(
  // DenseMap<FModuleLike, decltype(ModuleSignalMappings::mappings)> mappingsMap;
  std::string mappingsJsonString;
  llvm::raw_string_ostream mappingsJsonStream(mappingsJsonString);
  json::OStream mj(mappingsJsonStream, /* indentSize */ 2);
  // TODO: add fields we already know to mj

  auto gatherMappings = [](auto & mappings) {
    for (auto &m: mappings) {
    };
  };

    auto mb = OpBuilder::atBlockEnd(circuit.getBody());
    auto mJsonOp = mb.create<sv::VerbatimOp>(mb.getUnknownLoc(), mappingsJsonString);
    mJsonOp->setAttr(
        "output_file",
        hw::OutputFileAttr::getFromFilename(
          mb.getContext(), Twine(circuitPackage) + ".sigdrive.json", true));

  auto processModule = [this](FModuleOp module) -> Result {
    ModuleSignalMappings mapper(module, markDut, prefix);
    mapper.run();
    // XXX: HACK
    //assert(!mappingsMap.count(module));
    //if (!mapper.mappings.empty())
    //  mappingsMap[module] = mapper.mappings;

    return {mapper.allAnalysesPreserved,
            DenseMap<FModuleOp, DenseSet<unsigned>>(
                {{module, mapper.forcedInputPorts}})};
  };

  SmallVector<FModuleOp> modules;
  struct {
    // External modules put in the vsrcs field of the JSON.
    SmallVector<FExtModuleOp> vsrc;
    // External modules put in the "load_jsons" field of the JSON.
    SmallVector<FExtModuleOp> json;
  } extmodules;

  for (auto op : circuit.body().getOps<FModuleLike>()) {
    if (auto *extModule = dyn_cast<FExtModuleOp>(&op)) {
      AnnotationSet annotations(*extModule);
      if (annotations.hasAnnotation("firrtl.transforms.BlackBoxInlineAnno")) {
        extmodules.vsrc.push_back(*extModule);
        continue;
      }
      extmodules.json.push_back(*extModule);
      continue;
    }
    modules.push_back(cast<FModuleOp>(op));
  }

  auto reduce = [](const Result &acc, Result result) -> Result {
    DenseMap<FModuleOp, DenseSet<unsigned>> foo = acc.forcedInputPorts;
    foo.insert(result.forcedInputPorts.begin(), result.forcedInputPorts.end());
    return {acc.allAnalysesPreserved && result.allAnalysesPreserved, foo};
  };

  // XXX: BAD: TODO: FIXME: workaround for testing
  auto serialTransformReduce = [](auto Begin, auto End, auto Init, auto Reduce, auto Transform) {
    for (auto I = Begin; I != End; ++I)
      Init = Reduce(std::move(Init), Transform(*I));
    return std::move(Init);
  };

  // Note: this uses (unsigned)true instead of (bool)true for the reduction
  // because llvm::parallelTransformReduce uses the "data" method of std::vector
  // which is NOT provided for bool for optimization reasons.
  // Result result = llvm::parallelTransformReduce(
  Result result = serialTransformReduce(
      modules.begin(), modules.end(), Result(), reduce, processModule);

  auto *instanceGraph = &getAnalysis<InstanceGraph>();
  DenseMap<FModuleOp, ModuleNamespace> moduleNamespaces;
  for (auto fixup : result.forcedInputPorts) {
    for (auto portIdx : fixup.second) {
      for (auto *use : instanceGraph->lookup(fixup.first)->uses()) {
        auto inst = use->getInstance();
        auto port = inst->getResult(portIdx);
        OpBuilder builder(inst.getContext());
        builder.setInsertionPointAfter(inst);
        auto parentModule = inst->getParentOfType<FModuleOp>();
        ModuleNamespace &moduleNamespace = moduleNamespaces[parentModule];
        auto wire = builder.create<WireOp>(
            builder.getUnknownLoc(), port.getType(), builder.getStringAttr({}),
            builder.getArrayAttr({}),
            builder.getStringAttr(moduleNamespace.newName("_GEN")));
        port.replaceAllUsesWith(wire);
        builder.create<StrictConnectOp>(builder.getUnknownLoc(), port, wire);
      }
    }
  }

  if (result.allAnalysesPreserved)
    markAllAnalysesPreserved();

  // If this is a subcircuit, then continue on and emit JSON information
  // necessary to drive SiFive tools.
  if (!emitJSON)
    return;

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
