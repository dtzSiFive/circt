; RUN: firtool %s | FileCheck %s
FIRRTL version 4.0.0
circuit ForceVar:
  public module ForceVar:
     input clk : Clock
     input cond : UInt<1>
     output x : RWProbe<UInt<1>>

     wire w : UInt<1>
     w <= UInt(0)
     define x = rwprobe(w)

; CHECK:      reg break_net;
; CHECK-NEXT: break_net = ~ForceVar.w
; CHECK-NEXT: force ForceVar.w = break_net
     force_var(clk, cond, x, not(read(x)))

