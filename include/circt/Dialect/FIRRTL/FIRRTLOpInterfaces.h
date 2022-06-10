//===- FIRRTLOpInterfaces.h - Declare FIRRTL op interfaces ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the FIRRTL IR and supporting
// types.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_OP_INTERFACES_H
#define CIRCT_DIALECT_FIRRTL_OP_INTERFACES_H

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/CommandLine.h"

namespace circt {
namespace firrtl {

class FIRRTLType;

/// This holds the name and type that describes the module's ports.
struct PortInfo {
  StringAttr name;
  FIRRTLType type;
  Direction direction;
  StringAttr sym = {};
  Location loc = UnknownLoc::get(type.getContext());
  AnnotationSet annotations = AnnotationSet(type.getContext());

  StringRef getName() const { return name ? name.getValue() : ""; }

  /// Return true if this is a simple output-only port.  If you want the
  /// direction of the port, use the \p direction parameter.
  bool isOutput() {
    auto flags = type.getRecursiveTypeProperties();
    return flags.isPassive && !flags.containsAnalog &&
           direction == Direction::Out;
  }

  /// Return true if this is a simple input-only port.  If you want the
  /// direction of the port, use the \p direction parameter.
  bool isInput() {
    auto flags = type.getRecursiveTypeProperties();
    return flags.isPassive && !flags.containsAnalog &&
           direction == Direction::In;
  }

  /// Return true if this is an inout port.  This will be true if the port
  /// contains either bi-directional signals or analog types.
  bool isInOut() { return !isOutput() && !isInput(); }
};

/// Verification hook for verifying module like operations.
LogicalResult verifyModuleLikeOpInterface(FModuleLike module);

namespace detail {
  LogicalResult verifyInnerRefs(Operation *op);
} // namespace detail

/// Table of inner_sym's
class InnerSymbolTable {
public:
  /// Return the name of the attribute used for inner symbol names.
  static StringRef getInnerSymbolAttrName() { return "inner_sym"; }

  /// Build an inner_sym table for the given operation.
  InnerSymbolTable(Operation *op);

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists. Names never include the @ on them.
  Operation *lookup(StringRef name) const;
  template <typename T>
  T lookup(StringRef name) const {
    return dyn_cast_or_null<T>(lookup(name));
  }

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists. Names never include the @ on them.
  Operation *lookup(StringAttr name) const;
  template <typename T>
  T lookup(StringAttr name) const {
    return dyn_cast_or_null<T>(lookup(name));
  }

private:
  /// This is the operation this table is constructed for.
  /// Must have InnerSymbolTable trait.
  Operation *innerSymTblOp;

  /// This maps names to operations with that inner symbol.
  DenseMap<StringAttr, Operation *> symbolTable;
};

class InnerSymbolTableCollection {
public:
  // Get or create the InnerSymbolTable for the specified operation
  InnerSymbolTable & getInnerSymbolTable(Operation *op);

  void constructTablesInParallelFor(ArrayRef<Operation*> ops);
private:

  /// This maps Operations to their InnnerSymbolTables, constructed lazily
  DenseMap<Operation*, std::unique_ptr<InnerSymbolTable>> symbolTables;
};

class InnerRefNamespace {
public:
  SymbolTable &symTable;
  InnerSymbolTableCollection &innerSymTables;

  /// Resolve the InnerRef to the operation it targets.
  /// Returns null if no such name exists.
  /// Note that some InnerRef's target ports and must be handled separately.
  Operation *lookup(hw::InnerRefAttr inner);
  template <typename T>
  T lookup(hw::InnerRefAttr name) {
    return dyn_cast_or_null<T>(lookup(name));
  }
};

} // namespace firrtl
} // namespace circt

namespace mlir {
namespace OpTrait {

/// Scope for resolving innerref's
/// Must also be a SymbolTable.
/// Immediate children with the InnerSymbolTable trait will be resolved against.
template <typename ConcreteType>
class InnerRefNamespace : public TraitBase<ConcreteType, InnerRefNamespace> {
public:
  static LogicalResult verifyRegionTrait(Operation *op) {
    return ::circt::firrtl::detail::verifyInnerRefs(op);
  }
};

template <typename ConcreteType>
class InnerSymbolTable : public TraitBase<ConcreteType, InnerSymbolTable> {
public:
  static LogicalResult verifyRegionTrait(Operation *op) {
    // TODO: Walk looking for inner_syms, ensure they're unique ?

    // Must be nested within an op with InnerRefNamespace
    // For now, check the immediate parent has the trait.
    // We could also check `op->getParentWithTrait<...>()`
    // Don't check if 'op' has SymbolTable trait, although that's expected for innerref's
    auto *parent = op->getParentOp();
    return success(parent && parent->hasTrait<InnerRefNamespace>());
  }
};
} // namespace OpTrait
} // namespace mlir

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h.inc"
#endif // CIRCT_DIALECT_FIRRTL_OP_INTERFACES_H
