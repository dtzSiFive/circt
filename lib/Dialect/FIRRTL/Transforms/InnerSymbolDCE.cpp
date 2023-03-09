//===- InnerSymbolDCE.cpp - Delete Unused Inner Symbols----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass removes inner symbols which have no uses.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-inner-symbol-dce"

using namespace mlir;
using namespace circt;
using namespace firrtl;
using namespace hw;

struct InnerSymbolDCEPass : public InnerSymbolDCEBase<InnerSymbolDCEPass> {
  void runOnOperation() override;

private:
  void findInnerRefs(Attribute attr);
  void insertInnerRef(InnerRefAttr innerRef);
  void removeInnerSyms(HWModuleLike mod);

  DenseSet<std::pair<StringAttr, StringAttr>> innerRefs;
};

/// Find all InnerRefAttrs inside a given Attribute.
void InnerSymbolDCEPass::findInnerRefs(Attribute attr) {
  // Check if this Attribute or any sub-Attributes are InnerRefAttrs.
  attr.walk([&](Attribute subAttr) {
    if (auto innerRef = dyn_cast<InnerRefAttr>(subAttr))
      insertInnerRef(innerRef);
  });
}

/// Add an InnerRefAttr to the set of all InnerRefAttrs.
void InnerSymbolDCEPass::insertInnerRef(InnerRefAttr innerRef) {
  StringAttr moduleName = innerRef.getModule();
  StringAttr symName = innerRef.getName();

  // Track total inner refs found.
  ++numInnerRefsFound;

  auto [iter, inserted] = innerRefs.insert({moduleName, symName});
  if (!inserted)
    return;

  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": found reference to " << moduleName
                          << "::" << symName << '\n';);
}

/// Remove all dead inner symbols from the specified module.
void InnerSymbolDCEPass::removeInnerSyms(HWModuleLike mod) {
  auto moduleName = mod.moduleNameAttr();

  // Walk inner symbols, removing any not referenced.
  InnerSymbolTable::walkSymbols(
      mod, [&](StringAttr name, const InnerSymTarget &target) {
        ++numInnerSymbolsFound;

        // Check if the name is referenced by any InnerRef.
        if (innerRefs.contains({moduleName, name}))
          return;

        InnerSymbolTable::dropSymbol(target);
        ++numInnerSymbolsRemoved;

        LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": removed " << moduleName
                                << "::" << name << '\n';);
      });
}

void InnerSymbolDCEPass::runOnOperation() {
  // Run on the top-level ModuleOp to include any VerbatimOps that aren't
  // wrapped in a CircuitOp.
  ModuleOp topModule = getOperation();

  // Traverse the entire IR once.
  SmallVector<HWModuleLike> modules;
  topModule.walk([&](Operation *op) {
    // Find all InnerRefAttrs.
    for (NamedAttribute namedAttr : op->getAttrs())
      findInnerRefs(namedAttr.getValue());

    // Collect all HWModuleLike operations.
    if (auto mod = dyn_cast<HWModuleLike>(op))
      modules.push_back(mod);
  });

  // Traverse all FModuleOps in parallel, removing any InnerSymAttrs that are
  // dead code.
  parallelForEach(&getContext(), modules,
                  [&](HWModuleLike mod) { removeInnerSyms(mod); });
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInnerSymbolDCEPass() {
  return std::make_unique<InnerSymbolDCEPass>();
}
