//===- PrettyPrinterBuilder.h - Pretty printing builder -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper classes for using PrettyPrinter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PRETTYPRINTERBUILDER_H
#define CIRCT_SUPPORT_PRETTYPRINTERBUILDER_H

#include "circt/Support/PrettyPrinter.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace pretty {

//===----------------------------------------------------------------------===//
// Convenience builders.
//===----------------------------------------------------------------------===//

// Buffer tokens until EOF for clients that need to adjust things.
struct BufferingPP {
  using BufferVec = SmallVector<Token>;
  BufferVec &tokens;
  BufferingPP(BufferVec &tokens) : tokens(tokens) {}

  void add(Token t) { tokens.push_back(t); }

  /// Add a range of tokens.
  template <typename R>
  void addTokens(R &&tokens) {
    for (Token &t : tokens)
      add(t);
  }

  void eof() {}

  void flush(PrettyPrinter &pp) {
    pp.addTokens(tokens);
    tokens.clear();
  }
};

template <typename PPTy = PrettyPrinter>
class PPBuilder {
  PPTy &pp;

public:
  PPBuilder(PPTy &pp) : pp(pp) {}

  /// Add new token.
  template <typename T, typename... Args>
  typename std::enable_if_t<std::is_base_of_v<Token, T>> add(Args &&...args) {
    pp.add(T(std::forward<Args>(args)...));
  }
  void addToken(Token &t) { pp.add(t); }

  /// End of a stream.
  void eof() { pp.eof(); }

  /// Add a literal (with external storage).
  void literal(StringRef str) { add<StringToken>(str); }

  /// Add a non-breaking space.
  void nbsp() { literal(" "); }

  /// Add a newline (break too wide to fit, always breaks).
  void newline() { add<BreakToken>(PrettyPrinter::kInfinity); }

  /// End a group.
  void end() { add<EndToken>(); }

  /// Add breakable spaces.
  void spaces(uint32_t n) { add<BreakToken>(n); }

  /// Add a breakable space.
  void space() { spaces(1); }

  /// Add a break that is zero-wide if not broken.
  void zerobreak() { add<BreakToken>(0); }

  /// Start a consistent group with specified offset.
  void cbox(int32_t offset = 0, IndentStyle style = IndentStyle::Visual) {
    add<BeginToken>(offset, Breaks::Consistent, style);
  }

  /// Start an inconsistent group with specified offset.
  void ibox(int32_t offset = 0, IndentStyle style = IndentStyle::Visual) {
    add<BeginToken>(offset, Breaks::Inconsistent, style);
  }

  /// Open a cbox that closes when returned object goes out of scope.
  [[nodiscard]] auto scopedCBox(int32_t offset = 0,
                                IndentStyle style = IndentStyle::Visual) {
    cbox(offset, style);
    return llvm::make_scope_exit([&]() { end(); });
  }

  /// Open an ibox that closes when returned object goes out of scope.
  [[nodiscard]] auto scopedIBox(int32_t offset = 0,
                                IndentStyle style = IndentStyle::Visual) {
    ibox(offset, style);
    return llvm::make_scope_exit([&]() { end(); });
  }
};

/// PrettyPrinter::Listener that saves strings while live.
/// Once they're no longer referenced, memory is reset.
/// Allows differentiating between strings to save and external strings.
class PPBuilderStringSaver : public PrettyPrinter::Listener {
  llvm::BumpPtrAllocator alloc;
  llvm::StringSaver strings;

public:
  PPBuilderStringSaver() : strings(alloc) {}

  /// Add string, save in storage.
  StringRef save(StringRef str) { return strings.save(str); }

  /// PrettyPrinter::Listener::clear -- indicates no external refs.
  void clear() override;
};

//===----------------------------------------------------------------------===//
// Streaming support.
//===----------------------------------------------------------------------===//

/// Send one of these to PPStream to add the corresponding token.
/// See PPBuilder for details of each.
enum class PP {
  space,
  nbsp,
  newline,
  ibox0,
  ibox2,
  cbox0,
  cbox2,
  end,
  zerobreak,
  eof
};

/// String wrapper to indicate string has external storage.
struct PPExtString {
  StringRef str;
  explicit PPExtString(StringRef str) : str(str) {}
};

/// String wrapper to indicate string needs to be saved.
struct PPSaveString {
  StringRef str;
  explicit PPSaveString(StringRef str) : str(str) {}
};

/// Wrap a PrettyPrinter with PPBuilder features as well as operator<<'s.
/// String behavior:
/// Strings streamed as `const char *` are assumed to have external storage,
/// and StringRef's are saved until no longer needed.
/// Use PPExtString() and PPSaveString() wrappers to specify/override behavior.
template <typename PPTy = PrettyPrinter>
class PPStream : public PPBuilder<PPTy> {
  using Base = PPBuilder<PPTy>;
  PPBuilderStringSaver &saver;

public:
  /// Create a PPStream using the specified PrettyPrinter and StringSaver
  /// storage. Strings are saved in `saver`, which is generally the listener in
  /// the PrettyPrinter, but may not be (e.g., using BufferingPP).
  PPStream(PPTy &pp, PPBuilderStringSaver &saver) : Base(pp), saver(saver) {}

  /// Add a string literal (external storage).
  PPStream &operator<<(const char *s) {
    Base::literal(s);
    return *this;
  }
  /// Add a string token (saved to storage).
  PPStream &operator<<(StringRef s) {
    Base::template add<StringToken>(saver.save(s));
    return *this;
  }

  /// String has external storage.
  PPStream &operator<<(const PPExtString &str) {
    Base::literal(str.str);
    return *this;
  }

  /// String must be saved.
  PPStream &operator<<(const PPSaveString &str) {
    Base::template add<StringToken>(saver.save(str.str));
    return *this;
  }

  /// Convenience for inline streaming of builder methods.
  PPStream &operator<<(PP s) {
    switch (s) {
    case PP::space:
      Base::space();
      break;
    case PP::nbsp:
      Base::nbsp();
      break;
    case PP::newline:
      Base::newline();
      break;
    case PP::ibox0:
      Base::ibox();
      break;
    case PP::ibox2:
      Base::ibox(2);
      break;
    case PP::cbox0:
      Base::cbox();
      break;
    case PP::cbox2:
      Base::cbox(2);
      break;
    case PP::end:
      Base::end();
      break;
    case PP::zerobreak:
      Base::zerobreak();
      break;
    case PP::eof:
      Base::eof();
      break;
    }
    return *this;
  }

  /// Stream support for user-created Token's.
  PPStream &operator<<(Token &&t) {
    Base::addToken(t);
    return *this;
  }

  /// General-purpose "format this" helper, for types not supported by
  /// operator<< yet.
  template <typename T>
  void addAsString(T &&t) {
    *this << PPSaveString(llvm::formatv("{0}", t).str());
  }

  /// Helper to invoke code with a llvm::raw_ostream argument for compatibility.
  /// All data is gathered into a single string token.
  template <typename Callable>
  auto invokeWithStringOS(Callable &&C) {
    SmallString<128> ss;
    llvm::raw_svector_ostream ssos(ss);
    auto flush = llvm::make_scope_exit([&]() {
      if (!ss.empty())
        *this << ss;
    });
    return std::invoke(C, ssos);
  }

  /// Write escaped versions of the string, saved in storage.
  PPStream &writeEscaped(StringRef str, bool useHexEscapes = false) {
    return writeQuotedEscaped(str, useHexEscapes, "", "");
  }
  PPStream &writeQuotedEscaped(StringRef str, bool useHexEscapes = false,
                               StringRef left = "\"", StringRef right = "\"") {
    SmallString<64> ss;
    {
      llvm::raw_svector_ostream os(ss);
      os << left;
      os.write_escaped(str, useHexEscapes);
      os << right;
    }
    return *this << ss;
  }
};

} // end namespace pretty
} // end namespace circt

#endif // CIRCT_SUPPORT_PRETTYPRINTERBUILDER_H
