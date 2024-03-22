//===- IntrinsicInstancesToOps.cpp - Intmodule instance to ops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the IntrinsicInstancesToOps pass.  This pass processes
// FIRRTL intmodules and replaces all instances with generic intrinsic ops.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct IntrinsicInstancesToOpsPass
    : public IntrinsicInstancesToOpsBase<IntrinsicInstancesToOpsPass> {
  void runOnOperation() override;
  using IntrinsicInstancesToOpsBase::fixupEICGWrapper;
};
} // namespace

// This is the main entrypoint for the conversion pass.
void IntrinsicInstancesToOpsPass::runOnOperation() {
  auto &ig = getAnalysis<InstanceGraph>();

  // Convert to int ops.
  for (auto op :
       llvm::make_early_inc_range(getOperation().getOps<FIntModuleOp>())) {
    auto *node = ig.lookup(op);

    // Look at the intmodule's ports to determine how this gets converted.

    for (auto *use : node->uses()) {
      auto inst = use->getInstance<InstanceOp>();

      // Replace the instance of this intmodule with firrtl.int.generic.
      // Inputs become operands, outputs are the result (if any).
      ImplicitLocOpBuilder builder(op.getLoc(), inst);

      SmallVector<Value> inputs;
      struct OutputInfo {
        Value result;
        BundleType::BundleElement element;
      };
      SmallVector<OutputInfo> outputs;
      for (auto [idx, result] : llvm::enumerate(inst.getResults())) {
        // Replace inputs with wires that will be used as operands.
        if (inst.getPortDirection(idx) != Direction::Out) {
          auto w = builder.create<WireOp>(result.getLoc(), result.getType())
                       .getResult();
          result.replaceAllUsesWith(w);
          inputs.push_back(w);
          continue;
        }

        // Gather outputs.  This will become a bundle if more than one, but
        // typically there are zero or one.
        auto ftype = dyn_cast<FIRRTLBaseType>(inst.getType(idx));
        if (!ftype) {
          inst.emitError("intrinsic has non-FIRRTL or non-base port type")
              << inst.getType(idx);
          signalPassFailure();
          return;
        }
        outputs.push_back(
            OutputInfo{inst.getResult(idx),
                       BundleType::BundleElement(inst.getPortName(idx),
                                                 /*isFlipped=*/false, ftype)});
      }

      // Create the replacement operation.
      if (outputs.empty()) {
        // If no outputs, just create the operation.
        builder.create<GenericIntrinsicOp>(/*result=*/Type(),
                                           op.getIntrinsicAttr(), inputs,
                                           op.getParameters());

      } else if (outputs.size() == 1) {
        // For single output, the result is the output.
        auto resultType = outputs.front().element.type;
        auto intop = builder.create<GenericIntrinsicOp>(
            resultType, op.getIntrinsicAttr(), inputs, op.getParameters());
        outputs.front().result.replaceAllUsesWith(intop.getResult());
        // auto name = builder.getStringAttr(inst.getInstanceName() + "_" +
        // outputs.front().element.name.strref()); intop->setAttr("name", name);
      } else {
        // For multiple outputs, create a bundle with fields for each output
        // and replace users with subfields.
        auto resultType = builder.getType<BundleType>(llvm::map_to_vector(
            outputs, [](const auto &info) { return info.element; }));
        auto intop = builder.create<GenericIntrinsicOp>(
            resultType, op.getIntrinsicAttr(), inputs, op.getParameters());
        for (auto &output : outputs)
          output.result.replaceAllUsesWith(builder.create<SubfieldOp>(
              intop.getResult(), output.element.name));
        // intop->setAttr("name", inst.getInstanceNameAttr());
      }
      inst.erase();
    }
    op.erase();
    // Can't erase from IG.
    // ig.erase(node);
  }
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass>
circt::firrtl::createIntrinsicInstancesToOpsPass(bool fixupEICGWrapper) {
  auto pass = std::make_unique<IntrinsicInstancesToOpsPass>();
  pass->fixupEICGWrapper = fixupEICGWrapper;
  return pass;
}
