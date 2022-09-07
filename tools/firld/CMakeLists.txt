set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(firld
  firld.cpp
)
llvm_update_compile_flags(firld)

# TODO: Trim these
target_link_libraries(firld PRIVATE
  CIRCTExportChiselInterface
  CIRCTExportVerilog
  CIRCTImportFIRFile
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTHWTransforms
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTSVTransforms
  CIRCTTransforms

  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRFuncDialect
  MLIRTransforms
)
