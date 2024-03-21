//===- FIRRTLIntrinsics.cpp - Lower Intrinsics ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"

using namespace circt;
using namespace firrtl;

IntrinsicConverter::~IntrinsicConverter() = default;

ParseResult IntrinsicConverter::hasNInputs(unsigned n) {
  if (op.getNumOperands() != n)
    return op.emitError(name)
           << " has " << op.getNumOperands() << " inputs instead of " << n;
  return success();
}

// ParseResult IntrinsicConverter::namedPort(unsigned n, StringRef portName) {
//   auto ports = mod.getPorts();
//   if (n >= ports.size()) {
//     mod.emitError(name) << " missing port " << n;
//     return failure();
//   }
//   if (!ports[n].getName().equals(portName)) {
//     mod.emitError(name) << " port " << n << " named '" << ports[n].getName()
//                         << "' instead of '" << portName << "'";
//     return failure();
//   }
//   return success();
// }

ParseResult IntrinsicConverter::hasNParam(unsigned n, unsigned c) {
  unsigned num = 0;
  if (op.getParameters())
    num = op.getParameters().size();
  if (num < n || num > n + c) {
    auto d = op.emitError(name) << " has " << num << " parameters instead of ";
    if (c == 0)
      d << n;
    else
      d << " between " << n << " and " << (n + c);
    return failure();
  }
  return success();
}

ParseResult IntrinsicConverter::namedParam(StringRef paramName, bool optional) {
  for (auto a : op.getParameters()) {
    auto param = cast<ParamDeclAttr>(a);
    if (param.getName().getValue().equals(paramName)) {
      if (isa<StringAttr>(param.getValue()))
        return success();

      return op.emitError(name) << " has parameter '" << param.getName()
                                << "' which should be a string but is not";
    }
  }
  if (optional)
    return success();
  return op.emitError(name) << " is missing parameter " << paramName;
}

ParseResult IntrinsicConverter::namedIntParam(StringRef paramName,
                                              bool optional) {
  for (auto a : op.getParameters()) {
    auto param = cast<ParamDeclAttr>(a);
    if (param.getName().getValue().equals(paramName)) {
      if (isa<IntegerAttr>(param.getValue()))
        return success();

      return op.emitError(name) << " has parameter '" << param.getName()
                                << "' which should be an integer but is not";
    }
  }
  if (optional)
    return success();
  return op.emitError(name) << " is missing parameter " << paramName;
}

LogicalResult IntrinsicLowerings::lower(FModuleOp mod,
                                        bool allowUnknownIntrinsics) {
  unsigned numFailures = 0;

  PatternRewriter rewriter;

  mod.walk([](GenericIntrinsicOp intOp) {
    
  });
  for (auto op : llvm::make_early_inc_range(circuit.getOps<FModuleLike>())) {
    if (auto extMod = dyn_cast<FExtModuleOp>(*op)) {
      // Special-case some extmodules, identifying them by name.
      auto it = extmods.find(extMod.getDefnameAttr());
      if (it != extmods.end()) {
        if (succeeded(it->second(op))) {
          op.erase();
          ++numConverted;
        } else {
          ++numFailures;
        }
      }
      continue;
    }

    auto intMod = dyn_cast<FIntModuleOp>(*op);
    if (!intMod)
      continue;

    auto intname = intMod.getIntrinsicAttr();

    // Find the converter and apply it.
    auto it = intmods.find(intname);
    if (it == intmods.end()) {
      if (allowUnknownIntrinsics)
        continue;
      return op.emitError() << "intrinsic not recognized";
    }
    if (failed(it->second(op))) {
      ++numFailures;
      continue;
    }
    ++numConverted;
    op.erase();
  }

  return success(numFailures == 0);
}
