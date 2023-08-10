//===- ArcPasses.td - Arc dialect passes -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCPASSES_TD
#define CIRCT_DIALECT_ARC_ARCPASSES_TD

include "mlir/Pass/PassBase.td"

def AddTaps : Pass<"arc-add-taps", "::circt::hw::HWDesignOp"> {
  let summary = "Add taps to ports and wires such that they remain observable";
  let constructor = "circt::arc::createAddTapsPass()";
  let dependentDialects = ["arc::ArcDialect"];
  let options = [
    Option<"tapPorts", "ports", "bool", "true", "Make module ports observable">,
    Option<"tapWires", "wires", "bool", "true", "Make wires observable">,
    Option<"tapNamedValues", "named-values", "bool", "false",
           "Make values with `sv.namehint` observable">
  ];
}

def AllocateState : Pass<"arc-allocate-state", "arc::ModelOp"> {
  let summary = "Allocate and layout the global simulation state";
  let constructor = "circt::arc::createAllocateStatePass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def ArcCanonicalizer : Pass<"arc-canonicalizer", "::circt::hw::HWDesignOp"> {
  let summary = "Simulation centric canonicalizations";
  let constructor = "createArcCanonicalizerPass()";
  let dependentDialects = ["hw::HWDialect",
                           "comb::CombDialect",
                           "arc::ArcDialect"];
  let statistics = [
    Statistic<"numArcArgsRemoved", "num-arc-args-removed",
      "Number of arguments removed from DefineOps">,
  ];
}

def Dedup : Pass<"arc-dedup", "::circt::hw::HWDesignOp"> {
  let summary = "Deduplicate identical arc definitions";
  let description = [{
    This pass deduplicates identical arc definitions. If two arcs differ only by
    constants, the constants are outlined such that the arc can be deduplicated.
  }];
  let constructor = "circt::arc::createDedupPass()";
  let dependentDialects = ["arc::ArcDialect"];
  let statistics = [
    Statistic<"dedupPassNumArcsDeduped", "dedupPassNumArcsDeduped",
      "Number of arcs deduped">,
    Statistic<"dedupPassTotalOps", "dedupPassTotalOps",
      "Total number of ops deduped">,
  ];
}

def GroupResetsAndEnables : Pass<"arc-group-resets-and-enables",
                                 "::circt::hw::HWDesignOp"> {
  let summary = "Group reset and enable conditions of lowered states";
  let constructor = "circt::arc::createGroupResetsAndEnablesPass()";
  let dependentDialects = ["arc::ArcDialect", "mlir::scf::SCFDialect"];
}

def InferMemories : Pass<"arc-infer-memories", "::circt::hw::HWDesignOp"> {
  let summary = "Convert `FIRRTL_Memory` instances to dedicated memory ops";
  let constructor = "circt::arc::createInferMemoriesPass()";
  let dependentDialects = [
    "arc::ArcDialect", "comb::CombDialect", "seq::SeqDialect"
  ];
  let options = [
    Option<"tapPorts", "tap-ports", "bool", "true",
      "Make memory ports observable">,
  ];
}

def InlineArcs : Pass<"arc-inline" , "::circt::hw::HWDesignOp"> {
  let summary = "Inline very small arcs";
  let constructor = "circt::arc::createInlineArcsPass()";
  let statistics = [
    Statistic<"numInlinedArcs", "inlined-arcs", "Arcs inlined at a use site">,
    Statistic<"numRemovedArcs", "removed-arcs",
      "Arcs removed after full inlining">,
    Statistic<"numTrivialArcs", "trivial-arcs", "Arcs with very few ops">,
    Statistic<"numSingleUseArcs", "single-use-arcs", "Arcs with a single use">,
  ];
  let options = [
    Option<"intoArcsOnly", "into-arcs-only", "bool", "false",
           "Call operations to inline">,
    Option<"maxNonTrivialOpsInBody", "max-body-ops", "unsigned", "3",
           "Max number of non-trivial ops in the region to be inlined">,
  ];
}

def InlineModules : Pass<"arc-inline-modules", "::circt::hw::HWDesignOp"> {
  let summary = "Eagerly inline private modules";
  let description = [{
    This pass eagerly inlines private HW modules into their instantiation sites.
    After outlining combinational logic and registers into arcs, module bodies
    become fairly lightweight. Since arc definitions now fulfill the purpose of
    code reuse by allowing a single definition to be called multiple times, the
    module hierarchy degenerates into a purely cosmetic construct. At that point
    it is beneficial to fully flatten the module hierarchy to simplify further
    analysis and optimization of state transfer arcs.
  }];
  let constructor = "circt::arc::createInlineModulesPass()";
}

def InferStateProperties : Pass<"arc-infer-state-properties",
                                "::circt::hw::HWDesignOp"> {
  let summary = "Add resets and enables explicitly to the state operations";
  let constructor = "circt::arc::createInferStatePropertiesPass()";
  let dependentDialects = ["circt::hw::HWDialect", "circt::comb::CombDialect"];
}

def IsolateClocks : Pass<"arc-isolate-clocks", "::circt::hw::HWDesignOp"> {
  let summary = "Group clocked operations into clock domains";
  let constructor = "circt::arc::createIsolateClocksPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def LatencyRetiming : Pass<"arc-latency-retiming", "::circt::hw::HWDesignOp"> {
  let summary = "Push latencies through the design";
  let constructor = "circt::arc::createLatencyRetimingPass()";
  let dependentDialects = ["arc::ArcDialect"];

  let statistics = [
    Statistic<"numOpsRemoved", "num-ops-removed",
      "Number of zero-latency passthrough states removed">,
    Statistic<"latencyUnitsSaved", "latency-units-saved",
      "Number of latency units saved by merging them in a successor state">
  ];
}

def LegalizeStateUpdate : Pass<"arc-legalize-state-update", "::circt::hw::HWDesignOp"> {
  let summary = "Insert temporaries such that state reads don't see writes";
  let constructor = "circt::arc::createLegalizeStateUpdatePass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def LowerClocksToFuncs : Pass<"arc-lower-clocks-to-funcs", "::circt::hw::HWDesignOp"> {
  let summary = "Lower clock trees into functions";
  let constructor = "circt::arc::createLowerClocksToFuncsPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def LowerLUT : Pass<"arc-lower-lut", "arc::DefineOp"> {
  let summary = "Lowers arc.lut into a comb and hw only representation.";
  let constructor = "circt::arc::createLowerLUTPass()";
  let dependentDialects = ["hw::HWDialect", "comb::CombDialect"];
}

def LowerState : Pass<"arc-lower-state", "::circt::hw::HWDesignOp"> {
  let summary = "Split state into read and write ops grouped by clock tree";
  let constructor = "circt::arc::createLowerStatePass()";
  let dependentDialects = [
    "arc::ArcDialect", "mlir::scf::SCFDialect", "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def MakeTables : Pass<"arc-make-tables", "::circt::hw::HWDesignOp"> {
  let summary = "Transform appropriate arc logic into lookup tables";
  let constructor = "circt::arc::createMakeTablesPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def MuxToControlFlow : Pass<"arc-mux-to-control-flow", "::circt::hw::HWDesignOp"> {
  let summary = "Convert muxes with large independent fan-ins to if-statements";
  let constructor = "circt::arc::createMuxToControlFlowPass()";
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

def PrintStateInfo : Pass<"arc-print-state-info", "::circt::hw::HWDesignOp"> {
  let summary = "Print the state storage layout in JSON format";
  let constructor = "circt::arc::createPrintStateInfoPass()";
  let options = [
    Option<"stateFile", "state-file", "std::string", "",
      "Emit file with state description">
  ];
}

def SimplifyVariadicOps : Pass<"arc-simplify-variadic-ops", "::circt::hw::HWDesignOp"> {
  let summary = "Convert variadic ops into distributed binary ops";
  let constructor = "circt::arc::createSimplifyVariadicOpsPass()";
  let statistics = [
    Statistic<"numOpsSkippedMultipleBlocks", "skipped-multiple-blocks",
      "Ops skipped due to operands in different blocks">,
    Statistic<"numOpsSimplified", "simplified",
      "Ops simplified into binary ops">,
    Statistic<"numOpsCreated", "created",
      "Ops created as part of simplification">,
    Statistic<"numOpsReordered", "reordered",
      "Ops where simplification reordered operands">,
  ];
}

def SplitLoops : Pass<"arc-split-loops", "::circt::hw::HWDesignOp"> {
  let summary = "Split arcs to break zero latency loops";
  let constructor = "circt::arc::createSplitLoopsPass()";
  let dependentDialects = ["arc::ArcDialect"];
}

def StripSV : Pass<"arc-strip-sv", "::circt::hw::HWDesignOp"> {
  let summary = "Remove SV wire, reg, and assigns";
  let constructor = "circt::arc::createStripSVPass()";
  let dependentDialects = ["arc::ArcDialect", "comb::CombDialect",
                           "hw::HWDialect", "seq::SeqDialect"];
}

#endif // CIRCT_DIALECT_ARC_ARCPASSES_TD
