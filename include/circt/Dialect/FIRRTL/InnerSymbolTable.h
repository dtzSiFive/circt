//===- InnerSymbolTable.h - Inner Symbol Table -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the InnerSymbolTable and related classes, used for
// managing and tracking "inner symbols".
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_INNERSYMBOLTABLE_H
#define CIRCT_DIALECT_FIRRTL_INNERSYMBOLTABLE_H

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace firrtl {

/// The target of an inner symbol, the entity the symbol is a handle for.
class InnerSymTarget {
public:
  /// Default constructor, invalid.
  /* implicit */ InnerSymTarget() { assert(!*this); }

  /// Target an operation, and optionally a field (=0 means the op itself).
  explicit InnerSymTarget(Operation *op, size_t fieldID = 0)
      : op(op), portIdx(invalidPort), fieldID(fieldID) {}

  /// Target a port, and optionally a field (=0 means the port itself).
  /// Operation should be an FModuleLike.
  explicit InnerSymTarget(size_t portIdx, Operation *op, size_t fieldID = 0)
      : op(op), portIdx(portIdx), fieldID(fieldID) {}

  /// Create a target for a field, given a target to a base.
  explicit InnerSymTarget(const InnerSymTarget &base, size_t fieldID)
      : op(base.op), portIdx(base.portIdx), fieldID(fieldID) {
    assert(base.fieldID == 0);
  }

  InnerSymTarget(const InnerSymTarget &) = default;
  InnerSymTarget(InnerSymTarget &&) = default;

  // Accessors
  auto getField() { return fieldID; }
  Operation *getOp() { return op; }
  auto getPort() {
    assert(isPort());
    return portIdx;
  }

  // Classification
  bool isField() { return fieldID != 0; }
  bool isPort() { return portIdx != invalidPort; }
  bool isOpOnly() { return !isPort() && !isField(); }

private:
  auto asTuple() const { return std::tie(op, portIdx, fieldID); }
  Operation *op = nullptr;
  size_t portIdx = 0;
  size_t fieldID = 0;
  static constexpr size_t invalidPort = ~size_t{0};

public:
  // Comparison operators
  bool operator<(const InnerSymTarget &rhs) const {
    return asTuple() < rhs.asTuple();
  }
  bool operator==(const InnerSymTarget &rhs) const {
    return asTuple() == rhs.asTuple();
  }

  // Assignment
  InnerSymTarget &operator=(InnerSymTarget &&) = default;
  InnerSymTarget &operator=(const InnerSymTarget &) = default;

  // All targets must involve a valid op.
  operator bool() const { return op; }
};

/// A table of inner symbols and their resolutions.
class InnerSymbolTable {
public:
  /// Build an inner symbol table for the given operation.  The operation must
  /// have the InnerSymbolTable trait.
  explicit InnerSymbolTable(Operation *op);

  /// Non-copyable
  InnerSymbolTable(const InnerSymbolTable &) = delete;
  InnerSymbolTable &operator=(InnerSymbolTable &) = delete;

  /// Look up a symbol with the specified name, returning empty InnerSymTarget
  /// if no such name exists. Names never include the @ on them.
  InnerSymTarget lookup(StringRef name) const;
  InnerSymTarget lookup(StringAttr name) const;

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists or doesn't target just an operation.
  Operation *lookupOp(StringRef name) const;
  template <typename T>
  T lookupOp(StringRef name) const {
    return dyn_cast_or_null<T>(lookupOp(name));
  }

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists or doesn't target just an operation.
  Operation *lookupOp(StringAttr name) const;
  template <typename T>
  T lookupOp(StringAttr name) const {
    return dyn_cast_or_null<T>(lookupOp(name));
  }

  /// Get InnerSymbol for an operation.
  static StringAttr getInnerSymbol(Operation *op);

  /// Get InnerSymbol for a target.
  static StringAttr getInnerSymbol(InnerSymTarget target);

  /// Return the name of the attribute used for inner symbol names.
  static StringRef getInnerSymbolAttrName() { return "inner_sym"; }

private:
  /// This is the operation this table is constructed for, which must have the
  /// InnerSymbolTable trait.
  Operation *innerSymTblOp;

  /// This maps names to operations with that inner symbol.
  DenseMap<StringAttr, InnerSymTarget> symbolTable;
};

/// This class represents a collection of InnerSymbolTable's.
class InnerSymbolTableCollection {
public:
  /// Get or create the InnerSymbolTable for the specified operation.
  InnerSymbolTable &getInnerSymbolTable(Operation *op);

  /// Populate tables in parallel for all InnerSymbolTable operations in the
  /// given InnerRefNamespace operation.
  void populateTables(Operation *innerRefNSOp);

  explicit InnerSymbolTableCollection() = default;
  InnerSymbolTableCollection(const InnerSymbolTableCollection &) = delete;
  InnerSymbolTableCollection &operator=(InnerSymbolTableCollection &) = delete;

private:
  /// This maps Operations to their InnnerSymbolTable's.
  DenseMap<Operation *, std::unique_ptr<InnerSymbolTable>> symbolTables;
};

/// This class represents the namespace in which InnerRef's can be resolved.
struct InnerRefNamespace {
  SymbolTable &symTable;
  InnerSymbolTableCollection &innerSymTables;

  /// Resolve the InnerRef to its target within this namespace, returning empty
  /// target if no such name exists.
  InnerSymTarget lookup(hw::InnerRefAttr inner);

  /// Resolve the InnerRef to its target within this namespace, returning
  /// empty target if no such name exists or it's not an operation.
  /// Template type can be used to limit results to specified op type.
  Operation *lookupOp(hw::InnerRefAttr inner);
  template <typename T>
  T lookupOp(hw::InnerRefAttr inner) {
    return dyn_cast_or_null<T>(lookupOp(inner));
  }
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_INNERSYMBOLTABLE_H
