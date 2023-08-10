//===- HWTranslations.cpp - HW Translations -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the HW dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"

using namespace mlir;

namespace {

class HWDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const override {
    return success(isa<circt::hw::HWDesignOp>(op));
  }
};

} // namespace

void circt::hw::registerHWDialectTranslation(DialectRegistry &registry) {
  registry.insert<hw::HWDialect>();
  registry.addExtension(+[](MLIRContext *ctx, hw::HWDialect *dialect) {
    dialect->addInterfaces<HWDialectLLVMIRTranslationInterface>();
  });
}

void circt::hw::registerHWDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerHWDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
