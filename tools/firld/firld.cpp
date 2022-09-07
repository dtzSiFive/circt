//===- firld.cpp - The firrtl linker --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prototype FIRRTL linker.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/IR/Threading.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

// TODO: Support reading in FIR files!

/// Allow the user to specify the input file format.  This can be used to
/// override the input, and can be used to specify ambiguous cases like standard
/// input.
// enum InputFormatKind { InputUnspecified, InputFIRFile, InputMLIRFile };

static cl::OptionCategory mainCategory("firld Options");

// static cl::opt<InputFormatKind> inputFormat(
//     "format", cl::desc("Specify input file format:"),
//     cl::values(
//         clEnumValN(InputUnspecified, "autodetect", "Autodetect input
//         format"), clEnumValN(InputFIRFile, "fir", "Parse as .fir file"),
//         clEnumValN(InputMLIRFile, "mlir", "Parse as .mlir or .mlirbc file")),
//     cl::init(InputUnspecified), cl::cat(mainCategory));

static cl::list<std::string> inputFilenames(cl::Positional, cl::OneOrMore,
                                            cl::desc("<input files>"),
                                            cl::cat(mainCategory));

static cl::opt<std::string> outputFilename(
    "o", cl::desc("Output filename, or directory for split output"),
    cl::value_desc("filename"), cl::init("-"), cl::cat(mainCategory));

// TODO: useful for tests
// static cl::opt<bool>
//     verifyDiagnostics("verify-diagnostics",
//                       cl::desc("Check that emitted diagnostics match "
//                                "expected-* lines on the corresponding line"),
//                       cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

// static cl::opt<bool>
//     emitBytecode("emit-bytecode",
//                  cl::desc("Emit bytecode when generating MLIR output"),
//                  cl::init(false), cl::cat(mainCategory));

// static cl::opt<bool> force("f", cl::desc("Enable binary output on
// terminals"),
//                            cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> publicOnly("public-only",
                                cl::desc("Only print public symbols"),
                                cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> lowerAnnos("lower-annos",
                                cl::desc("Run lower annotations first"),
                                cl::init(false), cl::cat(mainCategory));

// This class prints logs before and after of pass executions. This
// insrumentation assumes that passes are not parallelized for firrtl::CircuitOp
// and mlir::ModuleOp.
class FirldPassInstrumentation : public mlir::PassInstrumentation {
  // This stores start time points of passes.
  using TimePoint = llvm::sys::TimePoint<>;
  llvm::SmallVector<TimePoint> timePoints;
  int level = 0;

public:
  void runBeforePass(Pass *pass, Operation *op) override {
    // This assumes that it is safe to log messages to stderr if the operation
    // is circuit or module op.
    if (isa<firrtl::CircuitOp, mlir::ModuleOp>(op)) {
      timePoints.push_back(TimePoint::clock::now());
      auto &os = llvm::errs();
      os << "[firld] ";
      os.indent(2 * level++);
      os << "Running \"";
      pass->printAsTextualPipeline(llvm::errs());
      os << "\"\n";
    }
  }

  void runAfterPass(Pass *pass, Operation *op) override {
    using namespace std::chrono;
    // This assumes that it is safe to log messages to stderr if the operation
    // is circuit or module op.
    if (isa<firrtl::CircuitOp, mlir::ModuleOp>(op)) {
      auto &os = llvm::errs();
      auto elapsed = duration<double>(TimePoint::clock::now() -
                                      timePoints.pop_back_val()) /
                     seconds(1);
      os << "[firld] ";
      os.indent(2 * --level);
      os << "-- Done in " << llvm::format("%.3f", elapsed) << " sec\n";
    }
  }
};

namespace {
struct InputFile {
  OwningOpRef<ModuleOp> mod;
  StringRef name;
};
struct FIRInputFile : public InputFile {
  firrtl::CircuitOp circt;
  // SymbolTable symtbl = SymbolTable(circt);
};
struct Entry {
  StringRef sym;
  Operation *op;
  SymbolTable::Visibility vis;
  InputFile *file;
  Entry(Operation *op, InputFile *file)
      : sym(SymbolTable::getSymbolName(op)), op(op),
        vis(SymbolTable::getSymbolVisibility(op)), file(file){};

  auto asTuple() const {
    return std::tie(sym, vis, file->name /* , op->getName() */);
  };
  bool operator<(const Entry &rhs) const { return asTuple() < rhs.asTuple(); }
  bool operator==(const Entry &rhs) const { return asTuple() == rhs.asTuple(); }
};
} // end anonymous namespace

/// This implements the top-level logic for the firld command, invoked once
/// command line options are parsed and LLVM/MLIR are all set up and ready to
/// go.
static LogicalResult executeFirld(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Register our dialects.
  context.loadDialect<firrtl::FIRRTLDialect, chirrtl::CHIRRTLDialect>();

  // For now:
  // Input fir or MLIR files.
  // Output: mlir file

  //===- Read inputs ------------------------------------------------------===//
  struct LeakyInputs {
    SmallVector<FIRInputFile> inputs;
    LeakyInputs(size_t n) { inputs.resize(n); }
    ~LeakyInputs() {
      // Leak
      for (auto &input : inputs)
        input.mod.release();
    };
  };
  auto numFiles = inputFilenames.size();
  LeakyInputs leakyInputs(numFiles);
  auto &inputs = leakyInputs.inputs;
  SourceMgr mainMgr;
  SourceMgrDiagnosticHandler handler(mainMgr, &context);
  {
    // Parse inputs into these structures in parallel, each with own SourceMgr.
    struct ParsedInput {
      StringRef name;
      OwningOpRef<ModuleOp> mod;
      SourceMgr mgr;
    };
    SmallVector<ParsedInput> srcs;
    srcs.resize(numFiles);
    auto parserTimer = ts.nest("Parsing inputs");
    auto loadFile = [&](size_t i) -> LogicalResult {
      auto &s = srcs[i];
      s.name = inputFilenames[i];
      auto fileParseTimer = parserTimer.nest(s.name);
      s.mod = parseSourceFile<ModuleOp>(s.name, s.mgr, &context);
      if (!s.mod)
        return failure();
      return success();
    };

    if (failed(failableParallelForEachN(&context, 0, numFiles, loadFile))) {
      llvm::errs() << "error reading inputs\n";
      return failure();
    }

    // Convert temporary ParsedInput to desired FIRFile's.
    for (auto [src, input]: llvm::zip(srcs, inputs)) {
       // Ensure handler's SourceMgr has buffers for this file.
       mainMgr.takeSourceBuffersFrom(src.mgr);

      // Check loaded module's are valid, create FIRFile from them.
      auto *body = src.mod->getBody();
      if (!body || !llvm::hasSingleElement(*body)) {
        // if (body) body->dump();
        if (body) {
          // sv.verbatim outside circuit :(
          for (auto &x : *body) {
            if (&x != &body->front())
              x.dump();
          }
        }
        return src.mod->emitError("must have body with single element");
      }
      auto circt = dyn_cast<firrtl::CircuitOp>(body->front());
      if (!circt)
        return body->front().emitError("expected circuit op");

      input = {{std::move(src.mod), src.name}, circt};
    }
  }

  //===- Lower annotations (optional) -------------------------------------===//
  if (lowerAnnos) {
    auto lowerTimer = ts.nest("Lowering Annotations");
    auto lowerAnnos = [&](auto &input) {
      // (construct once + reuse? combine under single 'module' and let PM
      // handle parallelism?)
      PassManager pm(&context);
      pm.enableVerifier(verifyPasses);
      pm.enableTiming(lowerTimer);
      if (verbosePassExecutions)
        pm.addInstrumentation(std::make_unique<FirldPassInstrumentation>());
      if (failed(applyPassManagerCLOptions(pm)))
        return failure();

      pm.nest<firrtl::CircuitOp>().addPass(
          firrtl::createLowerFIRRTLAnnotationsPass());

      if (failed(pm.run(input.mod.get())))
        return failure();
      return success();
    };
    // TODO: maybe put all in one module and let PM handle parallelism.
    if (failed(failableParallelForEach(&context, inputs, lowerAnnos))) {
      llvm::errs() << "error lowering annotations\n";
      return failure();
    }
  }

  //===- Grab symbols -----------------------------------------------------===//

  // TODO: String -> sym instead
  SmallVector<Entry> ents;
  ents.reserve(numFiles);
  for (auto &input : inputs) {
    llvm::errs() << llvm::formatv("Loaded {0}: {1}\n", input.name,
                                  input.circt.getName());
    // FIXME: redundant w/SymbolTable, but that doesn't expose iterators
    for (auto &op : *input.circt.getBodyBlock()) {
      assert(isa<SymbolOpInterface>(op));
      ents.emplace_back(&op, &input);
    }
  }

  llvm::sort(ents);

  auto publicEnts = llvm::make_filter_range(ents, [](const auto &e) {
    return e.vis == SymbolTable::Visibility::Public;
  });

  //===- Print entries ----------------------------------------------------===//
  auto print = [](auto &&range) {
    for (const auto &ent : range)
      llvm::errs() << llvm::formatv("{1,-8} {2,-18} {0}\t{3}\n", ent.sym,
                                    ent.vis, ent.op->getName(), ent.file->name);
  };

  if (publicOnly)
    print(publicEnts);
  else
    print(ents);

  llvm::errs() << "\n"; // help clang-format x.x

  //===- Magic------------------------------------------------------------===//

  for (auto &input: ArrayRef(inputs).drop_front()) {

  }

  // (I) Split IR interface, either to be linked back or in a new dialect that
  // "include"s definitions.

  // Foo.decl.ir:
  // circuit Foo:
  // module Foo:
  //   import Foo_Foo.mlir
  // module Bar
  //   import Foo_Bar.mlir

  // Also multiple-file designs composed in this way?

  // (I) Reference graph
  // Nodes:
  // * Ops
  //   * Symbol-defining ops
  //   * InnerSymbol-defining ops
  // Edges:
  // * Symbols and their uses (esp incl NLA's)
  //   * inst, NLA, ...

  // How to handle annotations?
  // * Store post-lowering?
  //   * lowering introduces verbatims, sv.interface's, etc.
  //   * GOAL: Represent entirely as FIR? Or represent annotations?
  // * Can't exactly link annotated IR in a general way--
  //   change hierarchy, instruct what passes to run where/when...
  //   * Inline / flatten
  //   * Prefix (linking concern)
  //   * ... many others

  // Annotations: belong as part of design (?)
  // Scope to component / TU.
  // * Worries about "top"
  // *

  // Claim: innersym visibility is not used/useful?
  // * Exploratory/counter: What /might/ it be good for re:cross-module refs?

  // sv.interface's are public, intended?


  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;

  // if (outputFormat != OutputSplitVerilog) {
  //   // Create an output file.
  //   outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  //   if (!outputFile.value()) {
  //     llvm::errs() << errorMessage << "\n";
  //     return failure();
  //   }
  // } else {
  //   // Create an output directory.
  //   if (outputFilename.isDefaultOption() || outputFilename == "-") {
  //     llvm::errs() << "missing output directory: specify with -o=<dir>\n";
  //     return failure();
  //   }
  //   auto error = llvm::sys::fs::create_directories(outputFilename);
  //   if (error) {
  //     llvm::errs() << "cannot create output directory '" << outputFilename
  //                  << "': " << error.message() << "\n";
  //     return failure();
  //   }
  // }

  // Process the input.
  // if (failed(processInput(context, ts, std::move(input), outputFile)))
  //   return failure();

  // If the result succeeded and we're emitting a file, close it.
  if (outputFile.has_value())
    outputFile.value()->keep();

  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register passes before parsing command-line options, so that they are
  // available for use with options like `--mlir-print-ir-before`.
  {
    // Dialect passes:
    firrtl::registerPasses();
    // sv::registerPasses();

    // Export passes:
    // registerExportChiselInterfacePass();
    // registerExportSplitChiselInterfacePass();
    // registerExportSplitVerilogPass();
    // registerExportVerilogPass();
  }

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();

  cl::AddExtraVersionPrinter(
      [](raw_ostream &os) { os << getCirctVersion() << '\n'; });
  cl::ParseCommandLineOptions(argc, argv, "MLIR-based FIRRTL linker\n");

  // Do the guts of the firld process.
  MLIRContext context;
  auto result = executeFirld(context);

  // Use "exit" instead of return to avoid costly `MLIRContext` teardown.
  exit(failed(result));
}
