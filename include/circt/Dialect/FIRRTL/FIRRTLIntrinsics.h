//===- FIRRTLIntrinsics.h - FIRRTL intrinsics ------------------*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers for the lowering of intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLINTRINSICS_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLINTRINSICS_H

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"

#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace firrtl {

/// Helper class for checking and extracting information from the generic instrinsic op.
struct GenericIntrinsic {
  GenericIntrinsicOp op;

  GenericIntrinsic(GenericIntrinsicOp op) : op(op) {}

  InFlightDiagnostic emitError() {
    return op.emitError(op.getIntrinsic());
  }

  ParseResult hasNInputs(unsigned n);

  // ParseResult hasOutput();

  // ParseResult hasBundleOutput();

  ParseResult hasNParam(unsigned n, unsigned c = 0);

  ParseResult namedParam(StringRef paramName, bool optional = false);

  ParseResult namedIntParam(StringRef paramName, bool optional = false);

  template <typename C>
  ParseResult checkInputType(unsigned n, const Twine &msg, C &&call) {
    if (n >= op.getNumOperands())
      return emitError() << " missing input " << n;
    if (!std::invoke(std::forward<C>(call), op.getOperand(n).getType()))
      return emitError() << " input " << n << " " << msg;
    return success();
  }

  template <typename C>
  ParseResult checkInputType(unsigned n, C &&call) {
    return checkInputType(n, "not of correct type", std::forward<C>(call));
  }

  template <typename T>
  ParseResult typedInput(unsigned n) {
    return checkInputType(n, [](auto ty) { return isa<T>(ty); });
  }

  ParseResult hasResetInput(unsigned n) {
    return checkInputType(n, "must be reset type", [](auto ty) {
      auto baseType = dyn_cast<FIRRTLBaseType>(ty);
      return baseType && baseType.isResetType();
    });
  }

  template <typename T>
  ParseResult sizedInput(unsigned n, int32_t size) {
    return checkInputType(n, "not size " + Twine(size), [size](auto ty) {
      auto t = dyn_cast<T>(ty);
      return t && t.getWidth() != size;
    });
  }

  template <typename T>
  ParseResult typedOutput() {
    if (op.getNumResults() == 0)
      return emitError() << " missing output";
    if (!isa<T>(op.getResult().getType()))
      return emitError() << " output not of correct type";
    return success();
  }

  template <typename T>
  ParseResult sizedOutput(int32_t size) {
    if (failed(typedOutput<T>()))
      return failure();
    if (cast<T>(op.getResult().getType()).getWidth() != size)
      return emitError() << " output not size " << size;
    return success();
  }
};

/// Base class for Intrinsic Converters.
///
/// Intrinsic converters contain validation logic, along with a converter
/// method to transform generic intrinsic ops to their implementation.
class IntrinsicConverter {
public:
  virtual ~IntrinsicConverter() = default;

  /// Checks whether the intrinsic is well-formed.
  ///
  /// This or's multiple ParseResults together, returning true on failure.
  virtual bool check(GenericIntrinsic gi) = 0;

  /// Transform the intrinsic to its implementation.
  virtual void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor, PatternRewriter &rewriter) = 0;
};

/// Lowering helper which collects all intrinsic converters.
class IntrinsicLowerings {
public:
  using ConversionMapTy =
      llvm::DenseMap<StringAttr, std::unique_ptr<IntrinsicConverter>>;

private:
  using ConverterFn = std::function<LogicalResult(GenericIntrinsicOp)>;

  /// Reference to the MLIR context.
  MLIRContext *context;

  /// Mapping from intrinsic names to converters.
  ConversionMapTy conversions;

public:
  IntrinsicLowerings(MLIRContext *context) : context(context) {}

  /// Registers a converter to an intrinsic name.
  // enable-if T is IntrinsicConverter.
  template <typename T>
  void add(StringRef name) {
    addConverter<T>(name);
  }

  /// Registers a converter to multiple intrinsic names.
  template <typename T, typename... Args>
  void add(StringRef name, Args... args) {
    add<T>(name);
    add<T>(args...);
  }

  /// Lowers all intrinsics in a module.
  LogicalResult lower(FModuleOp mod, bool allowUnknownIntrinsics = false);

private:
  template <typename T>
  void addConverter(StringRef name) {
    auto nameAttr = StringAttr::get(context, name);
    assert(!conversions.contains(nameAttr) && "duplicate conversion for intrinsic");
    conversions.try_emplace(nameAttr, std::make_unique<T>());
  }
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLINTRINSICS_H
