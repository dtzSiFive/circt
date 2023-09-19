//===- HoistPassthrough.cpp - Hoist basic passthrough ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the HoistPassthrough pass.  This pass identifies basic
// drivers of output ports that can be pulled out of modules.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWTypeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"

#include "llvm/ADT/GraphTraits.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PointerIntPair.h"

#include <deque>

#define DEBUG_TYPE "firrtl-hoist-passthrough"

using namespace circt;
using namespace firrtl;

using RefValue = mlir::TypedValue<RefType>;

namespace {

struct RefDriver;
struct HWDriver;

//===----------------------------------------------------------------------===//
// (Rematerializable)Driver declaration.
//===----------------------------------------------------------------------===//
/// Statically known driver for a Value.
///
/// Driver source expected to be rematerialized provided a mapping.
/// Generally takes form:
/// [source]----(static indexing?)---->DRIVE_OP---->[dest]
///
/// However, only requirement is that the "driver" can be rematerialized
/// across a module/instance boundary in terms of mapping args<-->results.
///
/// Driver can be reconstructed given a mapping in new location.
///
/// "Update":
/// Map:
///   source -> A
///   dest -> B
///
/// [source]---(indexing)--> SSA_DRIVE_OP ---> [dest]
///   + ([s']---> SSA_DRIVE_OP ---> [A])
///  =>
///  RAUW(B, [A]--(clone indexing))
///  (or RAUW(B, [s']--(clone indexing)))
///
/// Update is safe if driver classification is ""equivalent"" for each context
/// on the other side.  For hoisting U-Turns, this is safe in all cases,
/// for sinking n-turns the driver must be map-equivalent at all instantiation
/// sites.
/// Only UTurns are supported presently.
///
/// The goal is to drop the destination port, so after replacing all users
/// on other side of the instantiation, drop the port driver and move
/// all its users to the driver (immediate) source.
/// This may not be safe if the driver source does not dominate all users of the
/// port, in which case either reject (unsafe) or insert a temporary wire to
/// drive instead.
///
/// RAUW'ing may require insertion of conversion ops if types don't match.
//===----------------------------------------------------------------------===//
struct Driver {
  //-- Data -----------------------------------------------------------------//

  /// Connect entirely and definitively driving the destination.
  FConnectLike drivingConnect;
  /// Source of LHS.
  FieldRef source;

  //-- Constructors ---------------------------------------------------------//
  Driver(Value dest = nullptr) {}
  Driver(FConnectLike connect, FieldRef source)
      : drivingConnect(connect), source(source) {
    assert((isa<RefDriver, HWDriver>(*this)));
  }

  //-- Driver methods -------------------------------------------------------//

  // "Virtual" methods, either commonly defined or dispatched appropriately.

  /// Determine direct driver for the given value, empty Driver otherwise.
  static Driver get(Value v);

  /// Whether this can be rematerialized up through an instantiation.
  bool canHoist() const { return isa<BlockArgument>(source.getValue()); }

  /// Simple mapping across instantiation by index.
  using PortMappingFn = llvm::function_ref<Value(size_t)>;

  /// Rematerialize this driven value, using provided mapping function and
  /// builder. New value is returned.
  Value remat(PortMappingFn mapPortFn, ImplicitLocOpBuilder &builder);

  /// Drop uses of the destination, inserting temporary as necessary.
  /// Erases the driving connection, invalidating this Driver.
  void finalize(ImplicitLocOpBuilder &builder);

  //--- Helper methods -------------------------------------------------------//

  /// Return whether this driver is valid/non-null.
  operator bool() const { return source; }

  /// Get driven destination value.
  Value getDest() const {
    // (const cast to workaround getDest() not being const, even if mutates the
    // Operation* that's fine)
    return const_cast<Driver *>(this)->drivingConnect.getDest();
  }

  /// Whether this driver destination is a module port.
  bool drivesModuleArg() const {
    auto arg = dyn_cast<BlockArgument>(getDest());
    assert(!arg || isa<firrtl::FModuleLike>(arg.getOwner()->getParentOp()));
    return !!arg;
  }

  /// Whether this driver destination is an instance result.
  bool drivesInstanceResult() const {
    return getDest().getDefiningOp<hw::HWInstanceLike>();
  }

  /// Get destination as block argument.
  BlockArgument getDestBlockArg() const {
    assert(drivesModuleArg());
    return dyn_cast<BlockArgument>(getDest());
  }

  /// Get destination as operation result, must be instance result.
  OpResult getDestOpResult() const {
    assert(drivesInstanceResult());
    return dyn_cast<OpResult>(getDest());
  }

  /// Helper to obtain argument/result number of destination.
  /// Must be block arg or op result.
  static size_t getIndex(Value v) {
    if (auto arg = dyn_cast<BlockArgument>(v))
      return arg.getArgNumber();
    auto result = dyn_cast<OpResult>(v);
    assert(result);
    return result.getResultNumber();
  }
};

/// Driver implementation for probes.
struct RefDriver : public Driver {
  using Driver::Driver;

  static bool classof(const Driver *t) { return isa<RefValue>(t->getDest()); }

  static RefDriver get(Value v);

  Value remat(PortMappingFn mapPortFn, ImplicitLocOpBuilder &builder);
};
static_assert(sizeof(RefDriver) == sizeof(Driver),
              "passed by value, no slicing");

// Driver implementation for HW signals.
// Split out because has more complexity re:safety + updating.
// And can't walk through temporaries in same way.
struct HWDriver : public Driver {
  using Driver::Driver;

  static bool classof(const Driver *t) { return !isa<RefValue>(t->getDest()); }

  static HWDriver get(Value v);

  Value remat(PortMappingFn mapPortFn, ImplicitLocOpBuilder &builder);
};
static_assert(sizeof(HWDriver) == sizeof(Driver),
              "passed by value, no slicing");

/// Print driver information.
template <typename T>
static inline T &operator<<(T &os, Driver &d) {
  if (!d)
    return os << "(null)";
  return os << d.getDest() << " <-- " << d.drivingConnect << " <-- "
            << d.source.getValue() << "@" << d.source.getFieldID();
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Driver implementation.
//===----------------------------------------------------------------------===//

Driver Driver::get(Value v) {
  if (auto refDriver = RefDriver::get(v))
    return refDriver;
  if (auto hwDriver = HWDriver::get(v))
    return hwDriver;
  return {};
}

Value Driver::remat(PortMappingFn mapPortFn, ImplicitLocOpBuilder &builder) {
  return TypeSwitch<Driver *, Value>(this)
      .Case<RefDriver, HWDriver>(
          [&](auto *d) { return d->remat(mapPortFn, builder); })
      .Default({});
}

void Driver::finalize(ImplicitLocOpBuilder &builder) {
  auto immSource = drivingConnect.getSrc();
  auto dest = getDest();
  assert(immSource.getType() == dest.getType() &&
         "final connect must be strict");
  if (isa<BlockArgument>(immSource) || dest.hasOneUse()) {
    drivingConnect.erase();
    dest.replaceAllUsesWith(immSource);
  } else {
    // Insert wire temporary.
    // For hoisting use-case could also remat using cached indexing inside the
    // module, but wires keep this simple.
    auto temp = builder.create<WireOp>(immSource.getType());
    dest.replaceAllUsesWith(temp.getDataRaw());
  }
}

//===----------------------------------------------------------------------===//
// RefDriver implementation.
//===----------------------------------------------------------------------===//

static RefDefineOp getRefDefine(Value result) {
  for (auto *user : result.getUsers()) {
    if (auto rd = dyn_cast<RefDefineOp>(user); rd && rd.getDest() == result)
      return rd;
  }
  return {};
}

RefDriver RefDriver::get(Value v) {
  auto refVal = dyn_cast<RefValue>(v);
  if (!refVal)
    return {};

  auto rd = getRefDefine(v);
  if (!rd)
    return {};

  auto ref = getFieldRefFromValue(rd.getSrc(), true);
  if (!ref)
    return {};

  return RefDriver(rd, ref);
}

Value RefDriver::remat(PortMappingFn mapPortFn, ImplicitLocOpBuilder &builder) {
  auto mappedSource = mapPortFn(getIndex(source.getValue()));
  auto newVal = getValueByFieldID(builder, mappedSource, source.getFieldID());
  auto destType = getDest().getType();
  if (newVal.getType() != destType)
    newVal = builder.create<RefCastOp>(destType, newVal);
  return newVal;
}

//===----------------------------------------------------------------------===//
// HWDriver implementation.
//===----------------------------------------------------------------------===//

HWDriver HWDriver::get(Value v) {
  auto baseValue = dyn_cast<FIRRTLBaseValue>(v);
  if (!baseValue)
    return {};

  // Output must be passive, for flow reasons.
  // Reject aggregates for now, to be conservative re:aliasing writes/etc.
  // before ExpandWhens.
  if (!baseValue.getType().isPassive() || !baseValue.getType().isGround())
    return {};

  auto connect = getSingleConnectUserOf(v);
  if (!connect)
    return {};
  auto ref = getFieldRefFromValue(connect.getSrc());
  if (!ref)
    return {};

  // Reject if not all same block.
  if (v.getParentBlock() != ref.getValue().getParentBlock() ||
      v.getParentBlock() != connect->getBlock())
    return {};

  // Reject if cannot reason through this.
  if (hasDontTouch(v) || hasDontTouch(ref.getValue()))
    return {};
  if (auto fop = ref.getValue().getDefiningOp<Forceable>();
      fop && fop.isForceable())
    return {};

  // Limit to passive sources for now.
  auto sourceType = type_dyn_cast<FIRRTLBaseType>(ref.getValue().getType());
  if (!sourceType)
    return {};
  if (!sourceType.isPassive())
    return {};

  assert(hw::FieldIdImpl::getFinalTypeByFieldID(sourceType, ref.getFieldID()) ==
             baseValue.getType() &&
         "unexpected type mismatch, cast or extension?");

  return HWDriver(connect, ref);
}

Value HWDriver::remat(PortMappingFn mapPortFn, ImplicitLocOpBuilder &builder) {
  auto mappedSource = mapPortFn(getIndex(source.getValue()));
  // TODO: Cast if needed.  For now only support matching.
  // (No cast needed for current HWDriver's, getFieldRefFromValue and
  // assert)
  return getValueByFieldID(builder, mappedSource, source.getFieldID());
}

//===----------------------------------------------------------------------===//
// MustDrivenBy analysis.
//===----------------------------------------------------------------------===//
namespace {
/// Driver analysis, tracking values that "must be driven" by the specified
/// source (+fieldID), along with final complete driving connect.
class MustDrivenBy {
public:
  MustDrivenBy() = default;
  MustDrivenBy(FModuleOp mod) { run(mod); }

  /// Get direct driver, if computed, for the specified value.
  Driver getDriverFor(Value v) const { return driverMap.lookup(v); }

  /// Get combined driver for the specified value.
  /// Walks the driver "graph" from the value to its ultimate source.
  Driver getCombinedDriverFor(Value v) const {
    Driver driver = driverMap.lookup(v);
    if (!driver)
      return driver;

    // Chase and collapse.
    Driver cur = driver;
    size_t len = 1;
    SmallPtrSet<Value, 8> seen;
    while ((cur = driverMap.lookup(cur.source.getValue()))) {
      // If re-encounter same value, bail.
      if (!seen.insert(cur.source.getValue()).second)
        return {};
      driver.source = cur.source.getSubField(driver.source.getFieldID());
      ++len;
    }
    (void)len;
    llvm::dbgs() << "Found driver for " << v << " (chain length = " << len
                 << "): " << driver << "\n";
    return driver;
  }

  /// Analyze the given module's ports and chase simple storage.
  void run(FModuleOp mod) {
    SmallVector<Value, 64> worklist(mod.getArguments());

    DenseSet<Value> processed;
    processed.insert(worklist.begin(), worklist.end());
    while (!worklist.empty()) {
      auto val = worklist.pop_back_val();
      auto driver = ignoreHWDrivers ? RefDriver::get(val) : Driver::get(val);
      driverMap.insert({val, driver});
      if (!driver)
        continue;

      auto sourceVal = driver.source.getValue();

      // If already enqueued, ignore.
      if (!processed.insert(sourceVal).second)
        continue;

      // Only chase through atomic values for now.
      // Here, atomic implies must be driven entirely.
      // This is true for HW types, and is true for RefType's because
      // while they can be indexed into, only RHS can have indexing.
      if (hw::FieldIdImpl::getMaxFieldID(sourceVal.getType()) != 0)
        continue;

      // Only through Wires, block arguments, instance results.
      if (!isa<BlockArgument>(sourceVal) &&
          !isa_and_nonnull<WireOp, InstanceOp>(sourceVal.getDefiningOp()))
        continue;

      worklist.push_back(sourceVal);
    }
  }

  /// Clear out analysis results and storage.
  void clear() { driverMap.clear(); }

  /// Configure whether HW signals are analyzed.
  void setIgnoreHWDrivers(bool ignore) { ignoreHWDrivers = ignore; }

private:
  /// Map of values to their computed direct must-drive source.
  DenseMap<Value, Driver> driverMap;
  bool ignoreHWDrivers = false;
};
} // end anonymous namespace

namespace {
/// FieldRef cache.
class FieldRefs {
  /// Every block argument and result -> FieldRef.
  /// Built during walk, cached for re-use and easier querying.
  DenseMap<Value, FieldRef> valToFieldRef;

public:
  FieldRef getFor(Value v) const { return valToFieldRef.lookup(v); }
  FieldRef addRoot(Value v) {
    auto ref = FieldRef(v, 0);
    valToFieldRef.try_emplace(v, ref);
    return ref;
  }

  /// Add all results as roots.
  void addDecl(Operation *op) {
    for (auto result : op->getResults())
      addRoot(result);
  }
  /// Add argument as root.
  void addDecl(BlockArgument arg) { addRoot(arg); }

  /// Record derived value (indexing).
  FieldRef addDerived(Value input, Value derived, size_t fieldID) {
    assert(!valToFieldRef.contains(derived));
    auto inRef = getFor(input);
    assert(inRef);
    auto newRef = inRef.getSubField(fieldID);
    valToFieldRef.try_emplace(derived, newRef);
    return newRef;
  }

  /// Record subfield.
  FieldRef addIndex(SubfieldOp op) {
    auto access = op.getAccessedField();
    assert(op.getInput() == access.getValue());
    return addDerived(access.getValue(), op.getResult(), access.getFieldID());
  }

  /// Record subindex.
  FieldRef addIndex(SubindexOp op) {
    auto access = op.getAccessedField();
    assert(op.getInput() == access.getValue());
    return addDerived(access.getValue(), op.getResult(), access.getFieldID());
  }

  /// Record refsub.
  FieldRef addIndex(RefSubOp op) {
    auto input = op.getInput();
    auto inputBaseType = input.getType().getType();
    auto fieldID = hw::FieldIdImpl::getFieldID(inputBaseType, op.getIndex());
    return addDerived(input, op.getResult(), fieldID);
  }

  /// Clear storage.
  void clear() { valToFieldRef.clear(); }
};
} // end anonymous namespace

/// Whether the type can be atomically driven source -> dest.
static bool isAtomic(Type type) {
  // Skip foreign types for now, probably could be handled.
  if (!type_isa<FIRRTLType>(type))
    return false;
  // Refs, properties.
  if (hw::FieldIdImpl::getMaxFieldID(type) == 0)
    return true;
  // For HW, restrict to passive.
  FIRRTLBaseType baseType = type_dyn_cast<FIRRTLBaseType>(type);
  return baseType && baseType.isPassive();
}
[[maybe_unused]] static bool isAtomic(FieldRef ref) {
  return isAtomic(hw::FieldIdImpl::getFinalTypeByFieldID(
      ref.getValue().getType(), ref.getFieldID()));
}

namespace {
/// Simple connectivity graph.
struct ConnectionGraph {
  class Node {
  public:
    using NodeRef = Node *;
    /// Like a FieldRef but for a Node.
    /// FieldID is always on LHS, if RHS node is invalidated.
    using Edge = std::pair<NodeRef, size_t>;
  private:
    /// The definition represented by this node.
    /// Steal bit for state tracking (invalid).
    llvm::PointerIntPair<Value, 1, bool> defAndInvalid;
    /// Driver edges.  For now, track all but invalid if > 1.
    SmallVector<Edge, 1> drivenByEdges;

  public:
    Node(Value v) : defAndInvalid(v) {}

    void invalidate() { defAndInvalid.setInt(true); }
    bool isInvalid() const { return defAndInvalid.getInt(); }
    Value getDefinition() const { return defAndInvalid.getPointer(); }

    auto begin() { return drivenByEdges.begin(); }
    auto end() { return drivenByEdges.end(); }
    bool empty() { return drivenByEdges.empty(); }

    void addEdge(NodeRef node, size_t fieldID) {
      drivenByEdges.emplace_back(node, fieldID);
    }
  };
  using NodeRef = Node::NodeRef;
  using Edge = Node::Edge;

  // Lookup node for given value.
  // Node is basically value + edges, reconsider datastructure.
  DenseMap<Value, NodeRef> valToNode;

  // SpecificBumpPtrAllocator<Node>
  std::deque<Node> nodes;

  NodeRef lookup(Value v) const {
    return valToNode.lookup(v);
  }

  bool contains(Value v) const { return valToNode.contains(v); }

  NodeRef getOrCreateNode(Value v) {
    // Expensive sanity check.  Consider moving to an expensive-checks-only verify().
#ifndef NDEBUG
    auto ref = getFieldRefFromValue(v);
    if (ref.getValue() != v) {
      ref.getValue().dump();
      llvm::errs() << "fieldID: " << ref.getFieldID() << "\n";
      v.dump();
    }
    assert(ref.getValue() == v);
    assert(ref.getFieldID() == 0);
#endif
    auto [it, inserted] = valToNode.try_emplace(v, nullptr);
    if (!inserted)
      return it->second;
    nodes.emplace_back(v);
    return it->second = &nodes.back();
  };

  /// Add edge from src to dst.
  void addEdge(FieldRef src, Value v) {
    auto srcNode = getOrCreateNode(src.getValue());
    auto dstNode = getOrCreateNode(v);

    // Multiple drivers -> invalidate.
    // (if not already empty, it now has > 1)
    if (!dstNode->empty())
      dstNode->invalidate();
    dstNode->addEdge(srcNode, src.getFieldID());
  }
};

} // end anonymous namespace

namespace {

// TODO: Optionally disable analysis for HW signals!

class AtomicDriverAnalysis {
  /// Connectivity graph built by analysis.
  ConnectionGraph graph;
  /// Special dummy node inserted as "entry" to graph.
  /// "definition" is null value.
  ConnectionGraph::Node *modEntryNode;

  /// FieldRef cache, computed during analysis but cleared out when complete.
  FieldRefs refs;
public:
  ConnectionGraph &getGraph() { return graph; }
  ConnectionGraph::Node *getModEntryNode() { return modEntryNode; }

  auto nodes_begin() { return graph.nodes.begin(); }
  auto nodes_end() { return graph.nodes.end(); }

  AtomicDriverAnalysis(FModuleOp mod) {
    // Add dummy node.
    // TODO: Don't have null definition, set bit?.
    graph.nodes.emplace_back(Value());
    modEntryNode = &graph.nodes.back();

    run(mod);
  }

private:

  void invalidate(Value v) {
    return graph.getOrCreateNode(v)->invalidate();
  }

  void flow(FieldRef src, FieldRef dst) {
    // Non-root RHS invalidates the destination node.
    // We only want full connections.
    if (dst.getFieldID() != 0)
      return invalidate(dst.getValue());

    auto srcFType = type_dyn_cast<FIRRTLType>(src.getValue().getType());
    auto dstFType = type_dyn_cast<FIRRTLType>(dst.getValue().getType());
    assert(!dstFType || srcFType && "FIRRTL type driven by non-FIRRTL type?");

    auto srcBType = type_dyn_cast<FIRRTLBaseType>(srcFType);
    auto dstBType = type_dyn_cast<FIRRTLBaseType>(dstFType);
    if (srcBType) {
      assert(dstBType);
      // TODO: Maybe only portion of indexed type needs to be passive re:source.
      // Bundle of a and flip b, can say something is must-driven by 'a'.
      // For simplicity, only passive source and dest nodes for now.
      // (not only portions being connected)
      if (!srcBType.isPassive() || !dstBType.isPassive()) {
        invalidate(src.getValue());
        invalidate(dst.getValue());
        return;
      }
    } else if (type_isa<RefType>(srcFType)) {
      assert(type_isa<RefType>(dstFType));
    } else {
      // Everything else: invalidate source and destination, unhandled.
      invalidate(src.getValue());
      invalidate(dst.getValue());
      return;
    }

    return graph.addEdge(src, dst.getValue());
  }

  /// Add specified value as root, marking invalid as appropriate and only
  /// creating the node if needed to record invalid state.
  /// No need to create node for every thing unless involved in connectivity.
  void addLazyRoot(Value v, bool invalid = false) {
    refs.addRoot(v);
    if (invalid || !isAtomic(v.getType()) || hasDontTouch(v))
      graph.getOrCreateNode(v)->invalidate();
  }

  /// Add results of operation as root declarations. marking invalid as
  /// appropriate and only creating nodes if needed to record invalid state.
  void addDecl(Operation *op) {
    bool allInvalid = [&]() {
      if (hasDontTouch(op))
        return true;
      if (auto fop = dyn_cast<Forceable>(op); fop && fop.isForceable())
        return true;
      return false;
    }();

    for (auto result : op->getResults())
      addLazyRoot(result, allInvalid);
  }

  void run(FModuleOp mod) {
    /// Initialize with block arguments.
    for (auto arg : mod.getArguments()) {
      addLazyRoot(arg);
      /// Add output ports to dummy node, with "FieldID" as port number.
      if (mod.getPortDirection(arg.getArgNumber()) == Direction::Out)
        modEntryNode->addEdge(graph.getOrCreateNode(arg), arg.getArgNumber());
    }

    /// TODO: Use visitor!

    auto result =
        mod.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          auto result =
              TypeSwitch<Operation *, LogicalResult>(op)
                  .Case<SubfieldOp, SubindexOp, RefSubOp>([&](auto sub) {
                    assert(refs.getFor(sub.getInput()) &&
                           "indexing through unknown input");
                    refs.addIndex(sub);
                    return success();
                  })
                  .Case<SubaccessOp>([&](SubaccessOp access) {
                    invalidate(refs.getFor(access.getInput()).getValue());
                    addLazyRoot(access, true);
                    return success();
                  })
                  .Case<RefCastOp>([&](RefCastOp op) {
                    // Transparently index through refcast.
                    // TODO: Verification comparing to getFieldRefFromValue will disagree here!
                    refs.addDerived(op.getInput(), op.getResult(), 0);
                    return success();
                  })
                  .Case<NodeOp>([&](NodeOp node) {
                    addDecl(node);
                    auto inRef = refs.getFor(node.getInput());
                    assert(inRef);
                    flow(inRef, FieldRef(node.getResult(), 0));
                    return success();
                  })
                  .Case<Forceable, InstanceOp>([&](auto declOp) {
                    addDecl(declOp);
                    return success();
                  })
                  .Case<FConnectLike>([&](FConnectLike connect) {
                    auto srcRef = refs.getFor(connect.getSrc());
                    auto dstRef = refs.getFor(connect.getDest());
                    assert(srcRef);
                    assert(dstRef);
                    flow(srcRef, dstRef);

                    // Invalidate based on block containing connect and
                    // dest, based on connect "semantics".
                    if (!connect.hasStaticSingleConnectBehavior()) {
                      // Only support strict connect, and with all in same
                      // block. Conservative for now, can connect into when
                      // regions but not out.
                      if (!isa<StrictConnectOp>(connect) ||
                          connect->getBlock() !=
                              srcRef.getValue().getParentBlock() ||
                          connect->getBlock() !=
                              dstRef.getValue().getParentBlock())
                        invalidate(dstRef.getValue());
                    }
                    return success();
                  })
                  .Default([&](Operation *other) {
                    // Everything else treat as undriven root.
                    addDecl(other);

                    // Presently, analysis assumes unhandled operations are
                    // expressions -- with only 'connect' (through intermediate
                    // indexing ops) using as lvalue.

                    // TODO: Using visitor, support all expressions, and
                    // declarations generically. Anything else we don't know,
                    // mark all operands's roots as invalid. (?)

                    return success();
                  });
          return result;
        });

    // TODO: Either plumb this appropriately or drop it and what feeds it.
    (void)result;

     /// Clear out computed field refs, no longer needed.
     refs.clear();
  };
};


} // end anonymous namespace

/// Use a node as a "graph".  Useful for dfs, so on.
template <>
struct llvm::GraphTraits<ConnectionGraph::Node*> {
  using NodeType = ConnectionGraph::Node;
  using NodeRef = NodeType *;

  static NodeRef getEntryNode(NodeRef node) { return node; }

  static NodeRef getChild(const ConnectionGraph::Edge &edge) {
    return edge.first;
  }
  using EdgeIterator = std::invoke_result_t<decltype(&NodeType::begin),NodeType*>;
  using ChildIteratorType = llvm::mapped_iterator<EdgeIterator,decltype(&getChild)>;
  static ChildIteratorType child_begin(NodeRef node) {
    return {node->begin(), &getChild};
  }
  static ChildIteratorType child_end(NodeRef node) {
    return {node->end(), &getChild};
  }
};

/// Analysis as a graph, entry is dummy node "driven by" output ports.
template <>
struct llvm::GraphTraits<AtomicDriverAnalysis*> : public llvm::GraphTraits<ConnectionGraph::Node*> {
  static NodeRef getEntryNode(AtomicDriverAnalysis *graph) {
    return graph->getModEntryNode();
  }

  using node_inner_iterator = decltype(ConnectionGraph::nodes)::iterator;
  using nodes_iterator = llvm::pointer_iterator<node_inner_iterator>;
  static nodes_iterator nodes_begin(AtomicDriverAnalysis *graph) {
    return nodes_iterator(graph->getGraph().nodes.begin());
  }
  static nodes_iterator nodes_end(AtomicDriverAnalysis *graph) {
    return nodes_iterator(graph->getGraph().nodes.end());
  }
};

// Graph traits for DOT labeling.
template <>
struct llvm::DOTGraphTraits<AtomicDriverAnalysis *>
    : public llvm::DefaultDOTGraphTraits {
  using DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  static std::string getNodeLabel(const ConnectionGraph::Node *node,
                                  AtomicDriverAnalysis *ada) {
    if (node == ada->getModEntryNode()) {
      return "Entry dummy node";
    }
    assert(node);
    auto def = node->getDefinition();
    assert(def);
    // The name of the graph node is the module name.
    SmallString<128> str;
    llvm::raw_svector_ostream os(str);
    auto [name, valid] = getFieldName(FieldRef(def, 0), true);
    if (valid)
      os << name;
    else {
      // XXX: lmao.
      static mlir::AsmState asmState(def.getContext());
      def.print(os, asmState);
   }
    if (node->isInvalid())
      os << " INVALID";
    return os.str().str();
  }

  // Optionally, hide invalid nodes.  Comment out to toggle behavior.
#if 0
  static bool isNodeHidden(const ConnectionGraph::Node *node, const AtomicDriverAnalysis*) {
    return node->invalid;
  }
#endif

  // TODO: (optionally) Edge dest labels! (+invert)

  template <typename Iterator>
  static std::string getEdgeAttributes(const ConnectionGraph::Node *node, Iterator it,
                                       const AtomicDriverAnalysis *) {
    // Edge label is recorded fieldID (or argument number for edges from dummy entry).
    auto *cur = it.getCurrent();
    return ("label=" + Twine(cur->second)).str();
  }

  static std::string getGraphProperties(const AtomicDriverAnalysis*) {
    return "\trankdir=\"LR\";";
  }
};


//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {

struct HoistPassthroughPass
    : public HoistPassthroughBase<HoistPassthroughPass> {
  using HoistPassthroughBase::HoistPassthroughBase;
  void runOnOperation() override;

  using HoistPassthroughBase::hoistHWDrivers;
};
} // end anonymous namespace

void HoistPassthroughPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===- Running HoistPassthrough Pass "
                             "------------------------------------------===\n");
  auto &instanceGraph = getAnalysis<InstanceGraph>();

  SmallVector<FModuleOp, 0> modules(llvm::make_filter_range(
      llvm::map_range(
          llvm::post_order(&instanceGraph),
          [](auto *node) { return dyn_cast<FModuleOp>(*node->getModule()); }),
      [](auto module) { return module; }));

  MustDrivenBy driverAnalysis;
  driverAnalysis.setIgnoreHWDrivers(!hoistHWDrivers);

  // For each module (PO)...
  for (auto module : modules) {
    // TODO: Public means can't reason down into, or remove ports.
    // Does not mean cannot clone out wires or optimize w.r.t its contents.
    if (module.isPublic())
      continue;

    // 1. Analyze.

    // What ports to delete.
    // Hoisted drivers of output ports will be deleted.
    BitVector deadPorts(module.getNumPorts());

    // Analyze all ports using current IR.
    driverAnalysis.clear();
    driverAnalysis.run(module);

    LLVM_DEBUG(llvm::dbgs() << "Analyzing: " << module.getName() << "\n");
    if (true) {
      AtomicDriverAnalysis ada(module);

     // llvm::WriteGraph(&ada, module.getName());

      auto getSource = [&](ConnectionGraph::NodeRef node) -> FieldRef {
        // llvm::errs() << "Walking for: " << node->getDefinition() << "\n";
        FieldRef ref(node->getDefinition(), 0);
        for (auto I = llvm::df_begin(node), E = llvm::df_end(node); I != E; ++I) {
          // llvm::errs() << "\t" << I->getDefinition() << "\n";
          if (I->isInvalid())
            return {};

          // Search over.  Bail before inspecting edge below.
          if (I->empty()) {
            assert(std::next(I) == E);
            break;
          }
          // If multiple drivers, bail.
          if (!llvm::hasSingleElement(**I)) {
            assert(0 && "should be invalid or end if not single edge");
            return {};
          }
          auto &edge = *I->begin();
          if (I.nodeVisited(edge.first)) {
            mlir::emitRemark(node->getDefinition().getLoc(),
                             "driver cycle found")
                    .attachNote(edge.first->getDefinition().getLoc())
                << "already visited this value";
            return {};
          }
          // llvm::errs() << "\tIndex: " << edge.second << "\n";
          ref = FieldRef(edge.first->getDefinition(), edge.second)
                    .getSubField(ref.getFieldID());
        }
        // if (ref.getValue() == node->getDefinition())
        //   return {};
        return ref;
      };

      for (auto arg : module.getArguments()) {
        auto node = ada.getGraph().lookup(arg);
        if (!node)
          continue;
        auto source = getSource(node);
        if (source) {
          if (source.getValue() == arg) {
            // Input or undriven.
            LLVM_DEBUG(llvm::dbgs() << "self-source for : " << arg << "\n");
          } else
            LLVM_DEBUG(llvm::dbgs() << "Found driver for " << arg
                                    << " (chain length = TODO): "
                                    << "(no connect tracking)"
                                    << " source: " << source.getValue() << " @ "
                                    << source.getFieldID() << "\n");
        }
      }
    }

    auto notNullAndCanHoist = [](const Driver &d) -> bool {
      return d && d.canHoist();
    };


    SmallVector<Driver, 16> drivers(llvm::make_filter_range(
        llvm::map_range(module.getArguments(),
                        [&driverAnalysis](auto val) {
                          return driverAnalysis.getCombinedDriverFor(val);
                        }),
        notNullAndCanHoist));

    // 2. Rematerialize must-driven ports at instantiation sites.

    // Do this first, keep alive Driver state pointing to module.
    for (auto &driver : drivers) {
      std::optional<size_t> deadPort;
      {
        auto destArg = driver.getDestBlockArg();
        auto mod = cast<firrtl::FModuleLike>(destArg.getOwner()->getParentOp());
        auto index = destArg.getArgNumber();
        auto *igNode = instanceGraph.lookup(mod);

        // Replace dest in all instantiations.
        for (auto *record : igNode->uses()) {
          auto inst = cast<InstanceOp>(record->getInstance());
          ImplicitLocOpBuilder builder(inst.getLoc(), inst);
          builder.setInsertionPointAfter(inst);

          auto mappedDest = inst.getResult(index);
          mappedDest.replaceAllUsesWith(driver.remat(
              [&inst](size_t index) { return inst.getResult(index); },
              builder));
        }
        // The driven port has no external users, will soon be dead.
        deadPort = index;
      }
      assert(deadPort.has_value());

      assert(!deadPorts.test(*deadPort));
      deadPorts.set(*deadPort);

      // Update statistics.
      TypeSwitch<Driver *, void>(&driver)
          .Case<RefDriver>([&](auto *) { ++numRefDrivers; })
          .Case<HWDriver>([&](auto *) { ++numHWDrivers; });
    }

    // 3. Finalize stage.  Ensure remat'd dest is unused on original side.

    ImplicitLocOpBuilder builder(module.getLoc(), module.getBody());
    for (auto &driver : drivers) {
      // Finalize.  Invalidates the driver.
      builder.setLoc(driver.getDest().getLoc());
      driver.finalize(builder);
    }

    // If no ports were dropped, nothing to update.  Onwards!
    if (deadPorts.none())
      continue;

    // 4. Delete newly dead ports.

    // Drop dead ports at instantiation sites.
    auto *igNode = instanceGraph.lookup(module);
    for (auto *record : llvm::make_early_inc_range(igNode->uses())) {
      auto inst = cast<InstanceOp>(record->getInstance());
      ImplicitLocOpBuilder builder(inst.getLoc(), inst);

      assert(inst.getNumResults() == deadPorts.size());
      auto newInst = inst.erasePorts(builder, deadPorts);
      instanceGraph.replaceInstance(inst, newInst);
      inst.erase();
    }

    // Drop dead ports from module.
    module.erasePorts(deadPorts);

    numUTurnsHoisted += deadPorts.count();
  }
  markAnalysesPreserved<InstanceGraph>();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass>
circt::firrtl::createHoistPassthroughPass(bool hoistHWDrivers) {
  auto pass = std::make_unique<HoistPassthroughPass>();
  pass->hoistHWDrivers = hoistHWDrivers;
  return pass;
}
