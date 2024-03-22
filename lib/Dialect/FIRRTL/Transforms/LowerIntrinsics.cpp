//===- LowerIntrinsics.cpp - Lower Intrinsics -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerIntrinsics pass.  This pass processes FIRRTL
// extmodules with intrinsic annotations and rewrites the instances as
// appropriate.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
// #include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
// #include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
// #include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
// #include "llvm/ADT/APSInt.h"
// #include "llvm/ADT/StringMap.h"
// #include "llvm/ADT/PostOrderIterator.h"
// #include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace firrtl;

namespace {

class CirctSizeofConverter : public IntrinsicOpConverter<SizeOfIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedOutput<UIntType>(32) || gi.hasNParam(0);
  }
};

class CirctIsXConverter : public IntrinsicOpConverter<IsXIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedOutput<UIntType>(1) || gi.hasNParam(0);
  }
};

class CirctPlusArgTestConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(0) || gi.sizedOutput<UIntType>(1) || gi.hasNParam(1) ||
           gi.namedParam("FORMAT");
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    rewriter.replaceOpWithNewOp<PlusArgsTestIntrinsicOp>(
        gi.op, gi.getParamValue<StringAttr>("FORMAT"));
  }
};

class CirctPlusArgValueConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNOutputElements(2) ||
           gi.sizedOutputElement<UIntType>(0, "found", 1) ||
           gi.hasOutputElement(1, "result") || gi.hasNParam(1) ||
           gi.namedParam("FORMAT");
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto bty = gi.getOutputBundle().getType();
    auto newop = rewriter.create<PlusArgsValueIntrinsicOp>(
        gi.op.getLoc(), bty.getElementType(size_t{0}),
        bty.getElementType(size_t{1}), gi.getParamValue<StringAttr>("FORMAT"));
    rewriter.replaceOpWithNewOp<BundleCreateOp>(
        gi.op, bty, ValueRange({newop.getFound(), newop.getResult()}));
  }
};

class CirctClockGateConverter : public IntrinsicOpConverter<ClockGateIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    if (gi.op.getNumOperands() == 3) {
      return gi.typedInput<ClockType>(0) || gi.sizedInput<UIntType>(1, 1) ||
             gi.sizedInput<UIntType>(2, 1) || gi.typedOutput<ClockType>() ||
             gi.hasNParam(0);
    }
    if (gi.op.getNumOperands() == 2) {
      return gi.typedInput<ClockType>(0) || gi.sizedInput<UIntType>(1, 1) ||
             gi.typedOutput<ClockType>() || gi.hasNParam(0);
    }
    gi.emitError() << " has " << gi.op.getNumOperands()
                   << " ports instead of 3 or 4";
    return true;
  }
};

class CirctClockInverterConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.typedInput<ClockType>(0) ||
           gi.typedOutput<ClockType>() || gi.hasNParam(0);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    // TODO: As temporary accomodation, consider propagating name to op
    // during intmodule->op conversion, as code previously materialized
    // a wire to hold the name of the instance.
    // In the future, input FIRRTL can just be "node clock_inv = ....".
    rewriter.replaceOpWithNewOp<ClockInverterIntrinsicOp>(gi.op, adaptor.getOperands()[0]);

    // auto name = inst.getInstanceName();
    // Value outWire = builder.create<WireOp>(out.getType(), name).getResult();
    // builder.create<StrictConnectOp>(outWire, out);
    // inst.getResult(1).replaceAllUsesWith(outWire);
  }
};

class CirctMux2CellConverter : public IntrinsicOpConverter<Mux2CellIntrinsicOp> {
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(3) || gi.typedInput<UIntType>(0) || gi.hasNParam(0);
  }
};

class CirctMux4CellConverter
    : public IntrinsicOpConverter<Mux4CellIntrinsicOp> {
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(5) || gi.typedInput<UIntType>(0) || gi.hasNParam(0);
  }
};

class CirctLTLAndConverter : public IntrinsicOpConverter<LTLAndIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(2) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedInput<UIntType>(1, 1) || gi.sizedOutput<UIntType>(1) ||
           gi.hasNParam(0);
  }
};

class CirctLTLOrConverter : public IntrinsicOpConverter<LTLOrIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(2) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedInput<UIntType>(1, 1) || gi.sizedOutput<UIntType>(1) ||
           gi.hasNParam(0);
  }
};

} // namespace

#if 0

class CirctLTLDelayConverter : public IntrinsicConverter {
public:
  CirctLTLDelayConverter(StringRef name, FModuleLike mod)
      : IntrinsicConverter(name, mod) {
    auto getI64Attr = [&](int64_t value) {
      return IntegerAttr::get(IntegerType::get(mod.getContext(), 64), value);
    };

    auto params = mod.getParameters();
    delay = getI64Attr(params[0]
                           .cast<ParamDeclAttr>()
                           .getValue()
                           .cast<IntegerAttr>()
                           .getValue()
                           .getZExtValue());

    if (params.size() >= 2)
      if (auto lengthDecl = cast<ParamDeclAttr>(params[1]))
        length = getI64Attr(
            cast<IntegerAttr>(lengthDecl.getValue()).getValue().getZExtValue());
  }

  bool check() override {
    return hasNPorts(2) || namedPort(0, "in") || namedPort(1, "out") ||
           sizedPort<UIntType>(0, 1) || sizedPort<UIntType>(1, 1) ||
           hasNParam(1, 2) || namedIntParam("delay") ||
           namedIntParam("length", true);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    auto out =
        builder.create<LTLDelayIntrinsicOp>(in.getType(), in, delay, length);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }

private:
  IntegerAttr length;
  IntegerAttr delay;
};

class CirctLTLConcatConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "lhs") || namedPort(1, "rhs") ||
           namedPort(2, "out") || sizedPort<UIntType>(0, 1) ||
           sizedPort<UIntType>(1, 1) || sizedPort<UIntType>(2, 1) ||
           hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<LTLConcatIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLNotConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(2) || namedPort(0, "in") || namedPort(1, "out") ||
           sizedPort<UIntType>(0, 1) || sizedPort<UIntType>(1, 1) ||
           hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto input =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(input);
    auto out = builder.create<LTLNotIntrinsicOp>(input.getType(), input);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLImplicationConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "lhs") || namedPort(1, "rhs") ||
           namedPort(2, "out") || sizedPort<UIntType>(0, 1) ||
           sizedPort<UIntType>(1, 1) || sizedPort<UIntType>(2, 1) ||
           hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out =
        builder.create<LTLImplicationIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLEventuallyConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(2) || namedPort(0, "in") || namedPort(1, "out") ||
           sizedPort<UIntType>(0, 1) || sizedPort<UIntType>(1, 1) ||
           hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto input =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(input);
    auto out = builder.create<LTLEventuallyIntrinsicOp>(input.getType(), input);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLClockConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "in") || namedPort(1, "clock") ||
           namedPort(2, "out") || sizedPort<UIntType>(0, 1) ||
           typedPort<ClockType>(1) || sizedPort<UIntType>(2, 1) || hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto clock =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    inst.getResult(1).replaceAllUsesWith(clock);
    auto out = builder.create<LTLClockIntrinsicOp>(in.getType(), in, clock);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctLTLDisableConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "in") || namedPort(1, "condition") ||
           namedPort(2, "out") || sizedPort<UIntType>(0, 1) ||
           sizedPort<UIntType>(1, 1) || sizedPort<UIntType>(2, 1) ||
           hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto condition =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    inst.getResult(1).replaceAllUsesWith(condition);
    auto out =
        builder.create<LTLDisableIntrinsicOp>(in.getType(), in, condition);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

template <class Op>
class CirctVerifConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(1) || namedPort(0, "property") ||
           sizedPort<UIntType>(0, 1) || hasNParam(0, 1) ||
           namedParam("label", true);
  }

  LogicalResult convert(InstanceOp inst) override {
    auto params = mod.getParameters();
    StringAttr label;
    if (!params.empty())
      if (auto labelDecl = cast<ParamDeclAttr>(params[0]))
        label = cast<StringAttr>(labelDecl.getValue());

    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto property =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(property);
    builder.create<Op>(property, label);
    inst.erase();
    return success();
  }
};

class CirctHasBeenResetConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(3) || namedPort(0, "clock") || namedPort(1, "reset") ||
           namedPort(2, "out") || typedPort<ClockType>(0) || resetPort(1) ||
           sizedPort<UIntType>(2, 1) || hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto clock =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto reset =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(clock);
    inst.getResult(1).replaceAllUsesWith(reset);
    auto out = builder.create<HasBeenResetIntrinsicOp>(clock, reset);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
    return success();
  }
};

class CirctProbeConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return hasNPorts(2) || namedPort(0, "data") || namedPort(1, "clock") ||
           typedPort<ClockType>(1) || hasNParam(0);
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto clock =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto input =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(clock);
    inst.getResult(1).replaceAllUsesWith(input);
    builder.create<FPGAProbeIntrinsicOp>(clock, input);
    inst.erase();
    return success();
  }
};

} // namespace

// Replace range of values with new wires and return them.
template <typename R>
static SmallVector<Value> replaceResults(OpBuilder &b, R &&range) {
  return llvm::map_to_vector(range, [&b](auto v) {
    auto w = b.create<WireOp>(v.getLoc(), v.getType()).getResult();
    v.replaceAllUsesWith(w);
    return w;
  });
}

// Check ports are all inputs, emit diagnostic if not.
static ParseResult allInputs(ArrayRef<PortInfo> ports) {
  for (auto &p : ports) {
    if (p.direction != Direction::In)
      return mlir::emitError(p.loc, "expected input port");
  }
  return success();
}

// Get parameter by the given name.  Null if not found.
static ParamDeclAttr getNamedParam(ArrayAttr params, StringRef name) {
  for (auto param : params.getAsRange<ParamDeclAttr>())
    if (param.getName().getValue().equals(name))
      return param;
  return {};
}

namespace {

template <class OpTy, bool ifElseFatal = false>
class CirctAssertAssumeConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return namedPort(0, "clock") || typedPort<ClockType>(0) ||
           namedPort(1, "predicate") || sizedPort<UIntType>(1, 1) ||
           namedPort(2, "enable") || sizedPort<UIntType>(2, 1) ||
           namedParam("format", /*optional=*/true) ||
           namedParam("label", /*optional=*/true) ||
           namedParam("guards", /*optional=*/true) || allInputs(mod.getPorts());
    // TODO: Check all parameters accounted for.
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto params = mod.getParameters();
    auto format = getNamedParam(params, "format");
    auto label = getNamedParam(params, "label");
    auto guards = getNamedParam(params, "guards");

    auto wires = replaceResults(builder, inst.getResults());

    auto clock = wires[0];
    auto predicate = wires[1];
    auto enable = wires[2];

    auto substitutions = ArrayRef(wires).drop_front(3);
    auto name = label ? cast<StringAttr>(label.getValue()).strref() : "";
    // Message is not optional, so provide empty string if not present.
    auto message = format ? cast<StringAttr>(format.getValue())
                          : builder.getStringAttr("");
    auto op = builder.template create<OpTy>(clock, predicate, enable, message,
                                            substitutions, name,
                                            /*isConcurrent=*/true);
    if (guards) {
      SmallVector<StringRef> guardStrings;
      cast<StringAttr>(guards.getValue()).strref().split(guardStrings, ';');
      // TODO: Legalize / sanity-check?
      op->setAttr("guards", builder.getStrArrayAttr(guardStrings));
    }

    if constexpr (ifElseFatal)
      op->setAttr("format", builder.getStringAttr("ifElseFatal"));

    inst.erase();
    return success();
  }

private:
};

class CirctCoverConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check() override {
    return namedPort(0, "clock") || typedPort<ClockType>(0) ||
           namedPort(1, "predicate") || sizedPort<UIntType>(1, 1) ||
           namedPort(2, "enable") || sizedPort<UIntType>(2, 1) ||
           hasNPorts(3) || allInputs(mod.getPorts()) ||
           namedParam("label", /*optional=*/true) ||
           namedParam("guards", /*optional=*/true);
    // TODO: Check all parameters accounted for.
  }

  LogicalResult convert(InstanceOp inst) override {
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto params = mod.getParameters();
    auto label = getNamedParam(params, "label");
    auto guards = getNamedParam(params, "guards");

    auto wires = replaceResults(builder, inst.getResults());

    auto clock = wires[0];
    auto predicate = wires[1];
    auto enable = wires[2];

    auto name = label ? cast<StringAttr>(label.getValue()).strref() : "";

    // Empty message string for cover, only 'name' / label.
    auto op = builder.create<CoverOp>(clock, predicate, enable,
                                      builder.getStringAttr(""), ValueRange{},
                                      name, /*isConcurrent=*/true);
    if (guards) {
      SmallVector<StringRef> guardStrings;
      cast<StringAttr>(guards.getValue()).strref().split(guardStrings, ';');
      // TODO: Legalize / sanity-check?
      op->setAttr("guards", builder.getStrArrayAttr(guardStrings));
    }

    inst.erase();
    return success();
  }
};

} // namespace
#endif

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerIntrinsicsPass : public LowerIntrinsicsBase<LowerIntrinsicsPass> {
  void runOnOperation() override;
  // using LowerIntrinsicsBase::fixupEICGWrapper;
};
} // namespace

// TODO: Move to header or something?

// This is the main entrypoint for the lowering pass.
void LowerIntrinsicsPass::runOnOperation() {

  // TODO: Build conversion table/data-structure once in initialize().

  IntrinsicLowerings lowering(&getContext());
  lowering.add<CirctSizeofConverter>("circt.sizeof", "circt_sizeof");
  lowering.add<CirctIsXConverter>("circt.isX", "circt_isX");
  lowering.add<CirctPlusArgTestConverter>("circt.plusargs.test",
                                          "circt_plusargs_test");
  lowering.add<CirctPlusArgValueConverter>("circt.plusargs.value",
                                           "circt_plusargs_value");
  lowering.add<CirctClockGateConverter>("circt.clock_gate", "circt_clock_gate");
  lowering.add<CirctClockInverterConverter>("circt.clock_inv",
                                            "circt_clock_inv");
  lowering.add<CirctMux2CellConverter>("circt.mux2cell", "circt_mux2cell");
  lowering.add<CirctMux4CellConverter>("circt.mux4cell", "circt_mux4cell");

  lowering.add<CirctLTLAndConverter>("circt.ltl.and", "circt_ltl_and");
  lowering.add<CirctLTLOrConverter>("circt.ltl.or", "circt_ltl_or");

  if (failed(lowering.lower(getOperation(), /*allowUnknownIntrinsics=*/true)))
    return signalPassFailure();
  return;

  /// IntrinsicOpConversion::ConversionMapTy conversions;

  // auto &ig = getAnalysis<InstanceGraph>();

  // Rewrite firrtl.int.generic to specific intrinsic implementations.

#if 0
  IntrinsicLowerings lowering(&getContext(), ig);
  lowering.add<CirctSizeofConverter>("circt.sizeof", "circt_sizeof");
  lowering.add<CirctIsXConverter>("circt.isX", "circt_isX");
  lowering.add<CirctPlusArgTestConverter>("circt.plusargs.test",
                                          "circt_plusargs_test");
  lowering.add<CirctPlusArgValueConverter>("circt.plusargs.value",
                                           "circt_plusargs_value");
  lowering.add<CirctClockGateConverter>("circt.clock_gate", "circt_clock_gate");
  lowering.add<CirctClockInverterConverter>("circt.clock_inv",
                                            "circt_clock_inv");
  lowering.add<CirctLTLAndConverter>("circt.ltl.and", "circt_ltl_and");
  lowering.add<CirctLTLOrConverter>("circt.ltl.or", "circt_ltl_or");
  lowering.add<CirctLTLDelayConverter>("circt.ltl.delay", "circt_ltl_delay");
  lowering.add<CirctLTLConcatConverter>("circt.ltl.concat", "circt_ltl_concat");
  lowering.add<CirctLTLNotConverter>("circt.ltl.not", "circt_ltl_not");
  lowering.add<CirctLTLImplicationConverter>("circt.ltl.implication",
                                             "circt_ltl_implication");
  lowering.add<CirctLTLEventuallyConverter>("circt.ltl.eventually",
                                            "circt_ltl_eventually");
  lowering.add<CirctLTLClockConverter>("circt.ltl.clock", "circt_ltl_clock");
  lowering.add<CirctLTLDisableConverter>("circt.ltl.disable",
                                         "circt_ltl_disable");
  lowering.add<CirctVerifConverter<VerifAssertIntrinsicOp>>(
      "circt.verif.assert", "circt_verif_assert");
  lowering.add<CirctVerifConverter<VerifAssumeIntrinsicOp>>(
      "circt.verif.assume", "circt_verif_assume");
  lowering.add<CirctVerifConverter<VerifCoverIntrinsicOp>>("circt.verif.cover",
                                                           "circt_verif_cover");
  lowering.add<CirctMuxCellConverter<true>>("circt.mux2cell", "circt_mux2cell");
  lowering.add<CirctMuxCellConverter<false>>("circt.mux4cell",
                                             "circt_mux4cell");
  lowering.add<CirctHasBeenResetConverter>("circt.has_been_reset",
                                           "circt_has_been_reset");
  lowering.add<CirctProbeConverter>("circt.fpga_probe", "circt_fpga_probe");
  lowering.add<CirctAssertAssumeConverter<AssertOp>>(
      "circt.chisel_assert_assume", "circt_chisel_assert_assume");
  lowering.add<CirctAssertAssumeConverter<AssertOp, /*ifElseFatal=*/true>>(
      "circt.chisel_ifelsefatal", "circt_chisel_ifelsefatal");
  lowering.add<CirctAssertAssumeConverter<AssumeOp>>("circt.chisel_assume",
                                                     "circt_chisel_assume");
  lowering.add<CirctCoverConverter>("circt.chisel_cover", "circt_chisel_cover");

  // Remove this once `EICG_wrapper` is no longer special-cased by firtool.
  // if (fixupEICGWrapper)
  //   lowering.addExtmod<EICGWrapperToClockGateConverter>("EICG_wrapper");

  if (failed(lowering.lower(getOperation())))
    return signalPassFailure();
  if (!lowering.getNumConverted())
    markAllAnalysesPreserved();
#endif
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerIntrinsicsPass() {
  return std::make_unique<LowerIntrinsicsPass>();
}
