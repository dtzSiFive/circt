//===- HWDialect.h - HW dialect declaration ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an HW MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWDIALECT_H
#define CIRCT_DIALECT_HW_HWDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

// Pull in the dialect definition.
#include "circt/Dialect/HW/HWDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/HW/HWEnums.h.inc"

// TODO: Put this elsewhere?

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace circt {
namespace hw {
/// Register the HW dialect and the translation from it to the LLVM IR in
/// the given registry;
void registerHWDialectTranslation(mlir::DialectRegistry &registry);

/// Register the HW dialect and the translation from it in the registry
/// associated with the given context.
void registerHWDialectTranslation(mlir::MLIRContext &context);

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_HWDIALECT_H
