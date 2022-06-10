//===- InnerSymbolTable.cpp - InnerSymbolTable and InnerRef verification --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements InnerSymbolTable and verification for InnerRef's.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "mlir/IR/Threading.h"

using namespace circt;
using namespace firrtl;

namespace circt {
namespace firrtl {

//===----------------------------------------------------------------------===//
// InnerSymbolTable
//===----------------------------------------------------------------------===//

InnerSymbolTable::InnerSymbolTable(Operation *op) {
  assert(op->hasTrait<OpTrait::InnerSymbolTable>() &&
         "expected operation to have InnerSymbolTable trait");
  // TODO: relax these for, e.g., extmodule ?
  assert(op->getNumRegions() == 1 &&
         "expected operation to have a single region");
  assert(llvm::hasSingleElement(op->getRegion(0)) &&
         "expected operation to have a single block");

  // Save
  this->innerSymTblOp = op;


  // Build table

  // TODO: cache port lookups, in a sanely generic way? :(

  // if (auto mod = dyn_cast<FModuleLike>(op)) {
  //   for (auto p : llvm::enumerate(mod.getPorts()))
  //     symbolTable.insert({p.value().name, p.index())});
  // }

  // Add all operations
  // op->getRegion(0).walk([&](Operation *symOp) {
  op->walk([&](Operation *symOp) {
      auto attr = symOp->getAttrOfType<StringAttr>(InnerSymbolTable::getInnerSymbolAttrName());
      if (!attr)
        return;
      auto it = symbolTable.insert({attr, symOp});
      if (!it.second) {
        // TODO: obv this should be a diagnostic or something :)
        assert(0 && "repeated symbol found");
      }
  });
}

/// Look up a symbol with the specified name, returning null if no such name
/// exists. Names never include the @ on them.
Operation *InnerSymbolTable::lookup(StringRef name) const {
  return lookup(StringAttr::get(innerSymTblOp->getContext(), name));
}
Operation *InnerSymbolTable::lookup(StringAttr name) const {
  return symbolTable.lookup(name);
}

//===----------------------------------------------------------------------===//
// InnerSymbolTableCollection
//===----------------------------------------------------------------------===//

InnerSymbolTable &
InnerSymbolTableCollection::getInnerSymbolTable(Operation *op) {
  auto it = symbolTables.try_emplace(op, nullptr);
  if (it.second)
    it.first->second = ::std::make_unique<InnerSymbolTable>(op);
  return *it.first->second;
}

void InnerSymbolTableCollection::constructTablesInParallelFor(ArrayRef<Operation*> ops) {
  if (ops.empty())
    return;
  auto *ctx = ops[0]->getContext();

  // Ensure entries exist for each operation
  llvm::for_each(ops, [&](auto *op) { symbolTables.try_emplace(op, nullptr); });

  // Construct them in parallel
  mlir::parallelForEach(ctx, ops, [&](auto *op) {
      auto it = symbolTables.find(op);
      assert(it != symbolTables.end());
      it->second = ::std::make_unique<InnerSymbolTable>(op);
  });
}

//===----------------------------------------------------------------------===//
// InnerRefNamespace
//===----------------------------------------------------------------------===//

Operation *InnerRefNamespace::lookup(hw::InnerRefAttr inner) {
  auto mod = symTable.lookup(inner.getModule());
  assert(mod->hasTrait<mlir::OpTrait::InnerSymbolTable>());
  return innerSymTables.getInnerSymbolTable(mod).lookup(inner.getName());
}

//===----------------------------------------------------------------------===//
// InnerRef verification
//===----------------------------------------------------------------------===//

namespace detail {

LogicalResult verifyInnerRefs(Operation *op) {
  if (op->getNumRegions() != 1)
    return op->emitOpError() << "Operations with a 'InnerRefNamespace' must "
                                "have exactly one region";
  if (!llvm::hasSingleElement(op->getRegion(0)))
    return op->emitOpError() << "Operations with a 'InnerRefNamespace' must "
                                "have exactly one block";

  InnerSymbolTableCollection innerSymTables;
  // TODO: below, must be SymbolTable op (so our arg must be too)
  SymbolTable symbolTable(op);
  InnerRefNamespace ns{symbolTable,innerSymTables};

  // Gather top-level ops to process in parallel
  SmallVector<Operation *> childOps(
      llvm::make_pointer_range(op->getRegion(0).front()));
  // Filter these to those that have the InnerSymbolTable trait,
  // these will be walked in parallel to find all inner symbols.
  SmallVector<Operation *> innerSymTableOps(llvm::make_filter_range(childOps, [&](Operation *op) { return op->hasTrait<OpTrait::InnerSymbolTable>();}));
  innerSymTables.constructTablesInParallelFor(innerSymTableOps);

  auto verifySymbolUserFn = [&](Operation *op) -> WalkResult {
    if (auto user = dyn_cast<InnerRefUserOpInterface>(op))
      return WalkResult(user.verifyInnerRefs(ns));
    return WalkResult::advance();
  };
  return mlir::failableParallelForEach(op->getContext(), childOps, [&](auto *op) {
    return success(!op->walk(verifySymbolUserFn).wasInterrupted());
  });

  // For now, just walk everything and verify inner ref users
  // Parallelizing would be nice, but care re:lazy creation of InnerSymbolTables.
  // Controlling walk re:nested symboltables or innersymboltables, maybe?
  //WalkResult result =
  //  op->walk(verifySymbolUserFn);
  //    // walkSymbolTable(op->getRegions(), verifySymbolUserFn);
  //return success(!result.wasInterrupted());
}

} // namespace detail
} // namespace firrtl
} // namespace circt

