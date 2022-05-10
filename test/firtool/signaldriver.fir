  ; {
  ;   "class":"sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
  ;   "prefix":"prefix1_",
  ;   "inclusive":true,
  ;   "target":"~Foo|Bar"
  ; },
  ; {
  ;   "class":"sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
  ;   "prefix":"prefix2_",
  ;   "inclusive":true,
  ;   "target":"~Foo|Baz"
  ; }
; Source
      ;{
      ;  "_1":"~CoreIPSubsystemVerifTestHarness|BulletTile>clock",
      ;  "_2":"~signal_driver|signal_driver>clock_sink"
      ;},
; Sink
      ;{
      ;  "_1":"~CoreIPSubsystemVerifTestHarness|CoreIPSubsystem>clock",
      ;  "_2":"~signal_driver|signal_driver>reset_stretcher_cg_clock_source"
      ;}
circuit Foo: %[[
  {
  "class":"sifive.enterprise.grandcentral.SignalDriverAnnotation",
    "sinkTargets":[
      {
        "_1":"~Foo|Foo>i",
        "_2":"~signal_driver|signal_driver>Foo_i"
      },
      {
        "_1":"~Foo|Foo/b:Bar>i",
        "_2":"~signal_driver|signal_driver>Bar_i"
      }
    ],
    "sourceTargets":[
      {
        "_1":"~Foo|Foo>o",
        "_2":"~signal_driver|signal_driver>Foo_o"
      },
      {
        "_1":"~Bar|Bar>o",
        "_2":"~signal_driver|signal_driver>Bar_o"
      }
    ],
    "circuit":"circuit empty :\n  module empty :\n\n    skip\n",
    "annotations":[
    ],
    "circuitPackage":"foodriver"
}
]]
;circuit Foo:
  module Foo:
    input i: UInt<1>
    output o: UInt<1>
    inst b of Bar
    b.i <- i
    o <- b.o

  module Bar:
    input i: UInt<1>
    output o: UInt<1>
    o <- not(i)
