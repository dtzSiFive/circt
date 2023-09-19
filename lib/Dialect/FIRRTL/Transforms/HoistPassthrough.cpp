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
    LLVM_DEBUG(llvm::dbgs() << "Found driver for " << v << " (chain length = "
                            << len << "): " << driver << "\n");
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

/// FieldRef cache.
struct FieldRefs {
  /// Every block argument and result -> FieldRef.
  /// Built during walk, cached for re-use and easier querying.
  DenseMap<Value, FieldRef> valToFieldRef;

  FieldRef getFor(Value v) const {
    return valToFieldRef.lookup(v);
  }
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
  void addDecl(BlockArgument arg) {
    addRoot(arg);
  }
  FieldRef addDerived(Value input, Value derived, size_t fieldID) {
    assert(!valToFieldRef.contains(derived));
    auto inRef = getFor(input);
    assert(inRef);
    auto newRef = inRef.getSubField(fieldID);
    valToFieldRef.try_emplace(derived, newRef);
    return newRef;
  }

  FieldRef addIndex(SubfieldOp op) {
    auto access = op.getAccessedField();
    assert(op.getInput() == access.getValue());
    return addDerived(access.getValue(), op.getResult(), access.getFieldID());
  }
  FieldRef addIndex(SubindexOp op) {
    auto access = op.getAccessedField();
    assert(op.getInput() == access.getValue());
    return addDerived(access.getValue(), op.getResult(), access.getFieldID());
  }
  FieldRef addIndex(RefSubOp op) {
    auto input = op.getInput();
    auto fieldID = hw::FieldIdImpl::getFieldID(input.getType(), op.getIndex());
    return addDerived(input, op.getResult(), fieldID);
  }
};

/// Whether the type can be atomically driven source -> dest.
static bool isAtomic(Type type) {
  // Skip foreign types for now, probably could be handled.
  if (!type_isa<FIRRTLType>(type))
    return false;
  // Refs, properties.
  if (hw::FieldIdImpl::getMaxFieldID(type))
    return true;
  // For HW, restrict to passive.
  FIRRTLBaseType baseType = type_dyn_cast<FIRRTLBaseType>(type);
  return baseType && baseType.isPassive();
}
static bool isAtomic(FieldRef ref) {
  return isAtomic(hw::FieldIdImpl::getFinalTypeByFieldID(
      ref.getValue().getType(), ref.getFieldID()));
}

/// Simple connectivity graph.
struct ConnectionGraph {
  /// Nodes.
  struct Node;
  using NodeRef = Node*;
  /// Like a FieldRef but for a Node.
  /// FieldID is always on LHS, if RHS node is invalidated.
  using Edge = std::pair<NodeRef, size_t>;
  struct Node {
    Node(Value v) : definition(v) {}

    /// The definition represented by this node.
    Value definition;

    SmallVector<Edge, 1> drivenByEdges;
    // ConnectionGraph *parent;

    // TODO: PointerIntPair<Value,2,ClassificationEnum> or so.
    bool invalid = false;
    void invalidate() { invalid = true; }
  };

  // Lookup node for given value.
  // Node is basically value + edges, reconsider datastructure.
  DenseMap<Value, NodeRef> valToNode;

  // SpecificBumpPtrAllocator<Node>
  std::deque<Node> nodes;

  /// Entry nodes.
  SmallVector<NodeRef> entryNodes;

  // NodeRef getNode(FieldRef ref) const {
  //   auto it = nodeForRef.find(ref);
  //   if (it == nodeForRef.end()) {
  //     llvm::errs() << "Node not found for ref: " << ref.getValue() << " @ "
  //                  << ref.getFieldID() << "\n";
  //   }
  //   assert(it != nodeForRef.end());
  //   return it->second;
  // };
  // NodeRef getOrCreateNode(FieldRef ref) const {
  //   return nodeForLeafRef.lookup(ref);
  // };
  // NodeRef getOrCreateNode(FieldRef ref) {
  //   auto [it, inserted] = nodeForRef.try_emplace(ref, nullptr);
  //   if (!inserted)
  //     return it->second;
  //   nodes.emplace_back(ref);
  //   return it->second = &nodes.back();
  // };
  NodeRef getOrCreateNode(Value v) {
    // Expensive sanity check.  Consider moving to an expensive-checks-only verify().
#ifndef NDEBUG
    auto ref = getFieldRefFromValue(v);
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
    // auto srcNode = getNode(src.getValue());
    auto srcNode = getOrCreateNode(src.getValue());
    // auto srcNode = getOrCreateNode(src);
    auto dstNode = getOrCreateNode(v);
    // auto dstNode = getOrCreateNode(dst);

    dstNode->drivenByEdges.emplace_back(srcNode, src.getFieldID());
  }
};

} // end anonymous namespace

namespace {
class AtomicDriverAnalysis {
  ConnectionGraph graph;
  ConnectionGraph::Node modEntryNode = ConnectionGraph::Node(Value());
  FieldRefs refs;
public:
  ConnectionGraph &getGraph() { return graph; }
  const ConnectionGraph::Node *getModEntryNode() const { return &modEntryNode; }

  AtomicDriverAnalysis(FModuleOp mod) {
    run(mod);
    modEntryNode = ConnectionGraph::Node(Value());
    // TODO: Hoist entryNodes out, since we inject them ourselves anyway!
    // TODO: Don't have null definition :( .
    for (auto [idx, node] : llvm::enumerate(graph.entryNodes))
      modEntryNode.drivenByEdges.emplace_back(node, idx);
  }

private:

  void invalidate(Value v) {
    return graph.getOrCreateNode(v)->invalidate();
  }

  void flow(FieldRef src, FieldRef dst) {
    auto srcFType = type_dyn_cast<FIRRTLType>(src.getValue().getType());
    auto dstFType = type_dyn_cast<FIRRTLType>(dst.getValue().getType());
    if (!dstFType)
      return;
    assert(srcFType && "FIRRTL type driven by non-FIRRTL type?");

    // Non-root RHS invalidates the destination node.
    // We only want full connections.
    if (dst.getFieldID() != 0)
      return invalidate(dst.getValue());

    auto srcBType = type_dyn_cast<FIRRTLBaseType>(srcFType);
    auto dstBType = type_dyn_cast<FIRRTLBaseType>(dstFType);
    if (srcBType) {
      assert(dstBType);
      // TODO: Maybe only portion of indexed type needs to be passive re:source.
      // Bundle of a and flip b, can say something is must-driven by 'a'.
      // For simplicity, only passive nodes for now.
      if (!srcBType.isPassive() || !dstBType.isPassive()) {
        invalidate(src.getValue());
        invalidate(dst.getValue());
      }
    } else if (type_isa<RefType>(srcFType)) {
      assert(type_isa<RefType>(dstFType));
    }
    else
      return;

    return graph.addEdge(src, dst.getValue());
  }

  ConnectionGraph::NodeRef addRoot(Value v, bool invalid = false) {
    refs.addRoot(v);
    auto node = graph.getOrCreateNode(v);
    if (invalid || !isAtomic(v.getType()) || hasDontTouch(v))
      node->invalidate();
    return node;
  }

  /// Add results of operation as root declarations.
  /// Invalidate as appropriate.
  void addDecl(Operation *op) {
    bool allInvalid = [&]() {
      if (hasDontTouch(op))
        return true;
      if (auto fop = dyn_cast<Forceable>(op); fop && fop.isForceable())
        return true;
      return false;
    }();

    for (auto result : op->getResults())
      addRoot(result, allInvalid);
  }

  void run(FModuleOp mod) {
    /// Initialize with block arguments.

    for (auto arg : mod.getArguments()) {
      if (mod.getPortDirection(arg.getArgNumber()) == Direction::In) {
        addRoot(arg);
      } else {
        auto node = addRoot(arg);
        graph.entryNodes.push_back(node);
      }
    }

    /// TODO: Use visitor!

    auto result =
        mod.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          auto result =
              TypeSwitch<Operation *, LogicalResult>(op)
                  .Case<SubfieldOp, SubindexOp, RefSubOp>([&](auto sub) {
                    assert(refs.getFor(sub.getInput()) &&
                           "indexing through unknown input");
                    auto ref = refs.addIndex(sub);
                    return success();
                  })
                  .Case<SubaccessOp>([&](SubaccessOp access) {
                    invalidate(access.getInput());
                    addRoot(access, true);
                    return success();
                  })
                  .Case<RefCastOp>([&](RefCastOp op) {
                    // Transparently reference through refcast.
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
                    if (!srcRef || !dstRef) {
                      connect.getSrc().dump();
                      connect.getDest().dump();
                      connect.dump();
                    }
                    assert(srcRef);
                    assert(dstRef);
                    // connect.dump();
                    flow(srcRef, dstRef);

                    // Invalidate based on block containing connect and
                    // dest, based on connect "semantics".
                    if (!connect.hasStaticSingleConnectBehavior()) {
                      if (!isa<StrictConnectOp>(connect) ||
                          connect->getBlock() !=
                              srcRef.getValue().getParentBlock() ||
                          connect->getBlock() !=
                              dstRef.getValue().getParentBlock())
                        invalidate(dstRef.getValue());
                    }
                    if (!isa<StrictConnectOp,RefDefineOp>(connect)) {
                      invalidate(dstRef.getValue());
                    } else {
                    }
                    return success();
                  })
                  .Default([&](Operation *other) {
                    // Everything else treat as undriven root.
                    addDecl(other);

                    // TODO: Using visitor, support all expressions, and
                    // declarations. Anything else we don't know, mark all
                    // operands's roots as invalid.
                    return success();
                  });
          return result;
        });
    // return result;
    (void)result;
  };

};


} // end anonymous namespace

template <>
struct llvm::GraphTraits<AtomicDriverAnalysis*> {
  using NodeType = const ConnectionGraph::Node;
  using NodeRef = NodeType *;

  static NodeRef getEntryNode(AtomicDriverAnalysis *graph) {
    return graph->getModEntryNode();
  }

  static NodeRef getChild(const ConnectionGraph::Edge &edge) {
    return edge.first;
  }
  using EdgeIterator = decltype(NodeType::drivenByEdges)::const_iterator;
  using ChildIteratorType = llvm::mapped_iterator<EdgeIterator,decltype(&getChild)>;
  static ChildIteratorType child_begin(NodeRef node) {
    return {node->drivenByEdges.begin(), &getChild};
  }
  static ChildIteratorType child_end(NodeRef node) {
    return {node->drivenByEdges.end(), &getChild};
  }

  using node_inner_iterator = decltype(ConnectionGraph::nodes)::const_iterator;
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
                                  const AtomicDriverAnalysis *ada) {
    if (node == ada->getModEntryNode()) {
      return "Entry dummy node";
    }
    assert(node);
    assert(node->definition);
    // The name of the graph node is the module name.
    SmallString<128> str;
    llvm::raw_svector_ostream os(str);
    Value v = node->definition;
    v.print(os);
    return os.str().str();
  }

  // TODO: Edge dest labels!

  template <typename Iterator>
  static std::string getEdgeAttributes(const ConnectionGraph::Node *node, Iterator it,
                                       const AtomicDriverAnalysis *) {
    // Edge label is fieldID .
    return Twine(Twine("label=") + it->second).str();
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

    if (true) {
      AtomicDriverAnalysis ada(module);
     // for (auto &node : ada.getGraph().nodes) {
     //   llvm::errs() << (void *)&node << ":\n"
     //                << "\tval: " << node.definition << "\n\tinvalid? "
     //                << node.invalid << "\n\tdrivers ("
     //                << node.drivenByEdges.size() << "):\n";
     //   for (auto [node, fieldID] : node.drivenByEdges)
     //     llvm::errs() << "\t- (" << node->definition << ") @ " << fieldID
     //                  << "\n";
     // }
      llvm::WriteGraph(llvm::errs(), &ada);
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
