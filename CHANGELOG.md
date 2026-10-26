# Changelog

## [0.6.0](https://github.com/Luiz-Frias/entropic-substrate-lm/compare/v0.5.0...v0.6.0) (2026-10-25)


### Features

* **capture:** rebase D1MlflowObserver on MetricSink, add strict sink ([#376](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/376)) ([fa4f497](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/fa4f497893626d051bd1a893da3f88c00220219a))
* **run:** record N continuations with their sampling law (FOR-1674) ([#385](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/385)) ([411d310](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/411d31076ee56566fc0471c94240bbe716cc8d9d))
* **run:** record observed bf16 autocast in run_start (FOR-1672) ([#384](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/384)) ([5a94e7c](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/5a94e7c8832c54c482e1b806150caaa3f21d7a8e))


### Bug Fixes

* **run:** label zero-sentence mechanism rows absent (FOR-1673) ([#383](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/383)) ([ef09910](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ef09910e6761ba88b62d54fb5d3908baeee672cf))
* **run:** wave C review round 1 fixes and guard pins ([#386](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/386)) ([43dab35](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/43dab35e04ca1c152cbe075b196d178be8bac369))
* **scripts:** collect files the blind reviewer writes in its sandbox ([#382](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/382)) ([4a1d90b](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/4a1d90b27f3b14a76c9be80e4beffa60b7f042e4))


### Code Refactoring

* **cli:** move H and embedding handler families to siblings ([#377](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/377)) ([61fa546](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/61fa546080668abda2248ea3f0d6f572af24b3b3))


### Documentation

* **briefs:** wave B briefs FOR-1552 (strict observer), FOR-1608 ([#375](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/375)) ([863b74c](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/863b74c8d7b9e3573a7ed7f27eae07629bd8c1ef))
* **briefs:** wave C briefs FOR-1671, FOR-1672, FOR-1673, FOR-1674 ([#381](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/381)) ([9ad2538](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/9ad2538e9ca3a4e22fe9261aff8563724af4c0b3))
* **h1:** add the H1 preregistration skeleton for the preservation 2x2 ([#380](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/380)) ([40124cd](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/40124cd7da0d535080d82906b5bbffcb040ef0b2))

## [0.5.0](https://github.com/Luiz-Frias/entropic-substrate-lm/compare/v0.4.1...v0.5.0) (2026-10-23)


### Features

* **h-analysis:** add H0 final-probe checks and a resume builder ([#374](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/374)) ([128a3d2](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/128a3d27132513e69e3af51c237a70675979c9c6))


### Documentation

* **h0:** record blind review round 1 for wave HEpP7g ([#373](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/373)) ([0832dac](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/0832dacfcccb57c8267217b095d5c566dc94b7da))

## [0.4.1](https://github.com/Luiz-Frias/entropic-substrate-lm/compare/v0.4.0...v0.4.1) (2026-10-22)


### Bug Fixes

* **deploy:** re-pin the H image to the v0.4.0 tip ([#371](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/371)) ([70d2b5f](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/70d2b5f1fe37ac685291e2b81436b9e899b34bc0))

## [0.4.0](https://github.com/Luiz-Frias/entropic-substrate-lm/compare/v0.3.3...v0.4.0) (2026-10-22)


### Features

* **core:** key evaluation seed by step and add diagnostic samples ([#366](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/366)) ([7a113ac](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/7a113ac0d54abd743269dd8e4ae14b711e0b4c60))
* **domain:** memoize admitted evaluation records in seal memo ([#365](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/365)) ([868120b](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/868120b7dc986acf327d3654baaa959b38c52f93))
* **h-analysis:** FOR-1648 aggregate, charts and report over H runs ([#367](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/367)) ([d830197](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/d830197e7a22ff3bfca711d7ab3879e092fbdc43))
* **scripts:** make blind review model configurable via env ([#369](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/369)) ([50a4951](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/50a4951ea7094c5f63fb00d2e83a4f44432c9828))


### Bug Fixes

* **ladder:** alias D1 status enums to the kernel verdict vocab ([#362](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/362)) ([dd16ff5](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/dd16ff5da2cdfa05628d2432de2023be8ac2a691))


### Code Refactoring

* **d1:** drop dead manifest-diff loop in D1ExperimentFamily ([#361](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/361)) ([c2ac149](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/c2ac149cf27618fceea338134ccd6755e4346264))
* **scripts:** split assemble_review_packet under 1000 lines ([#360](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/360)) ([7333634](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/73336341b0135833cf8e5089ef2e56a81da7087d))


### Documentation

* **briefs:** add lane briefs for FOR-1670, FOR-1648 and FOR-1664 ([#364](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/364)) ([75ad2f2](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/75ad2f269888e5df963d796d9a04126290084e4a))
* **briefs:** bug-squash wave briefs FOR-1598, FOR-1634, FOR-1609 ([#359](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/359)) ([2d195a0](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/2d195a0b235002a1408d13508bbf8d1dbe2bd2f9))
* **h0:** add plumbing analysis for wave HEpP7g ([#368](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/368)) ([d3c8f46](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/d3c8f46745e8f9df7fd4dd5685e51af472b18db9))
* **h0:** add the HEpP7g capture pointer and analysis brief ([#363](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/363)) ([2f0e25a](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/2f0e25a860920160c9bebabb2803c912c25437a4))

## [0.3.3](https://github.com/Luiz-Frias/entropic-substrate-lm/compare/v0.3.2...v0.3.3) (2026-10-20)


### Bug Fixes

* **h-run:** FOR-1669 mirror final-leg stages to MLflow ([#357](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/357)) ([6ef75ec](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/6ef75ec66d39b374a39072570b8fbf1c1b0a0f92))

## [0.3.2](https://github.com/Luiz-Frias/entropic-substrate-lm/compare/v0.3.1...v0.3.2) (2026-10-19)


### Bug Fixes

* **deploy:** re-pin the H image to the v0.3.1 tip ([#355](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/355)) ([577f37c](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/577f37cef129a82c8aa1ff215319d97ab5dad4ce))

## [0.3.1](https://github.com/Luiz-Frias/entropic-substrate-lm/compare/v0.3.0...v0.3.1) (2026-10-19)


### Bug Fixes

* **h-host:** FOR-1665 defer provider absence on a transient probe read ([daa756d](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/daa756da0aca54a1385723b207aec5368f3e427e))
* **h-host:** FOR-1665 tolerate transient R2 error on terminal probe ([daa756d](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/daa756da0aca54a1385723b207aec5368f3e427e))
* **h-host:** FOR-1665 tolerate transient R2 error on terminal probe ([#351](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/351)) ([daa756d](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/daa756da0aca54a1385723b207aec5368f3e427e))
* **h-host:** FOR-1667 permanent D1 read codes fail fast on probe ([#353](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/353)) ([dc9c655](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/dc9c6558e4e23ed45b6a3c89be5a49cb4aa72f6a))
* **h-run:** FOR-1662 one admission walk per root; stages reuse the memo ([4b7f3bd](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/4b7f3bd3a4c8379cb5eb368794886fd9eb597a56))
* **h-run:** FOR-1662 one admission walk per root; stages reuse the memo ([#352](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/352)) ([4b7f3bd](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/4b7f3bd3a4c8379cb5eb368794886fd9eb597a56))


### Documentation

* **briefs:** FOR-1662 brief amended after the design gate, D1 to D5 ([4b7f3bd](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/4b7f3bd3a4c8379cb5eb368794886fd9eb597a56))

## [0.3.0](https://github.com/Luiz-Frias/entropic-substrate-lm/compare/v0.2.1...v0.3.0) (2026-10-17)


### Features

* **capture:** FOR-1639 H runs take their own capture lane ([#343](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/343)) ([fd88937](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/fd88937f5d4abf80db9c03773ceb721e0c41bdcc))
* **capture:** FOR-1642 H run surface, checkpoint evidence, FAILED ([#345](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/345)) ([68bae8c](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/68bae8c6434c0059b0e3b0ea51fb76fb8b0b4c00))
* **h-host:** host substrate, provider-aware controller, v3 launch ([#346](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/346)) ([c2c47d0](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/c2c47d0f0e0e68bf35f772b49fafc791a4d17405))
* **h-worker:** FOR-1657 tag every H run with its request digest ([#347](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/347)) ([dbe9709](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/dbe97094358ca349ce20ee1a396ae44d1c7700aa))


### Bug Fixes

* **capture:** FOR-1650 commit a capture pass only when all admissible ([#344](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/344)) ([c280a84](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/c280a8474752c1e0240b00e765498433d464b132))
* **deploy:** re-pin the H image to the observability parity tip ([#349](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/349)) ([8aec100](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/8aec1008d15db3ebb725424550a24f96627d7433))
* **logging:** FOR-1649 capsys tests assert on real log lines ([#342](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/342)) ([80575ea](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/80575eaf141e16a5f1576d68569e5cdbfcbd0ac3))
* **run:** H worker narrates every mechanism through the log substrate (FOR-1637) ([#341](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/341)) ([8b29b46](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/8b29b4642cbab96e6eade1dd7ec9fe9778fec87b))

## [0.2.1](https://github.com/Luiz-Frias/entropic-substrate-lm/compare/v0.2.0...v0.2.1) (2026-10-12)


### Bug Fixes

* **deploy:** re-pin the H image to the v0.2.0 release tip ([cb2d852](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/cb2d8522e6f87146c61c4d0ae50ab1c6b6e08aa4))
* **deploy:** re-pin the H image to the v0.2.0 release tip ([#338](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/338)) ([73f6bcd](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/73f6bcda870bb3f2f4fe40f61a6230a528e262c0))


### Documentation

* **plans:** add the refactor retrospective, correct the p14 register ([a34e8d2](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/a34e8d2dd19de5d61fd94a6235fbf5bdfa8c2921))
* **plans:** add the refactor retrospective, correct the p14 register ([#339](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/339)) ([7d7c375](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/7d7c3757b4a56b257b4d79f6b44273cfbc8e9bfe))

## [0.2.0](https://github.com/Luiz-Frias/entropic-substrate-lm/compare/v0.1.0...v0.2.0) (2026-10-12)


### Features

* **capture:** write MlflowSink metrics off the training thread ([4261f40](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/4261f404f5d6be404d99864a68e6f5fdfcad80ea))
* **capture:** write MlflowSink metrics off the training thread ([#312](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/312)) ([c1ed884](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/c1ed884052eeb63eafeed69fe97d03c1de732080))
* **codec:** add a parse_float seam to strict_loads (FOR-1632) ([#334](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/334)) ([008c179](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/008c17923edf0422f575b49a9029d8d7905eb89a))
* **codec:** add strict-JSON boundary codec (FOR-1613) ([2b2dc03](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/2b2dc03b35d768c10ded26ee5580acb12c81b68a))
* **codec:** add strict-JSON boundary codec (FOR-1613) ([#316](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/316)) ([e929910](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/e929910a5dd86f06f4d8910533a6ee19ef56010e))
* **run:** graceful operator stop for the H probe lane ([fb514fa](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/fb514faaa3ba4d0411c954f5c4314f60f0ae6576))
* **run:** graceful operator stop for the H probe lane ([#311](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/311)) ([adc4314](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/adc43148ebbce874c52b5030056098f21fb7823f))
* **verdict:** enforce D1 closed vocabularies as StrEnum (FOR-1573) ([fc7728d](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/fc7728d31ec4648fb5a8ada17a592603598cc6f0))


### Bug Fixes

* **analyze:** D1RunReader fails loud on gap metrics (FOR-1579) ([b7634a0](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/b7634a0f3ac655e52d3e95be4e558b2eba3fa00f))
* **analyze:** D1RunReader fails loud on gap metrics (FOR-1579) ([602b5a2](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/602b5a2e9e73813821ee4f3db3bb6503d22a78ee))
* **analyze:** D1RunReader fails loud on gap metrics (FOR-1579) ([#327](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/327)) ([b7634a0](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/b7634a0f3ac655e52d3e95be4e558b2eba3fa00f))
* **boundary:** repoint three SHA sites to the lowercase door (FOR-1626) ([#335](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/335)) ([58e52a3](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/58e52a37cf4dddb40e947a235fb42ebddf2c36d9))
* **capture:** count post-finish log() and force sync log_batch ([ec9aa9e](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ec9aa9ecfa62b6e47bbe7f2ee22bad0c3ae43673))
* **capture:** explicit writer stop; injectable finish() drain budget ([bdfb2aa](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/bdfb2aa8a83f60565c1e5b84a3b17312f3023f21))
* **capture:** never re-invoke create_run after a run_id exists ([7d3a0b4](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/7d3a0b49790a07ddf2e14db86901616b38ba5406))
* **capture:** never re-invoke create_run after a run_id exists ([#323](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/323)) ([bd6fae4](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/bd6fae4a75144988ca85806f5184ee24e5264fd3))
* **cli:** resolve handlers statically, drop _lazy (FOR-1571) ([3770de9](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/3770de941f6eb24266f52fdc33f8a4226bd80f25))
* **cli:** resolve handlers statically, drop _lazy (FOR-1571) ([a8f7381](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/a8f73815c3b7f9a41ec3430294d13fff0491ee06))
* **codec:** reject non-finite floats in strict_loads; scope fail ([e27ba87](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/e27ba8760167f8e43fd8a637865001de2fd11efe))
* **historical:** clean a failed remote worker workspace (FOR-1580) ([ef38fdf](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ef38fdf653a815baa7aa74d153daded4ae6f3bd4))
* **historical:** clean a failed remote worker workspace (FOR-1580) ([ac76ba9](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ac76ba95bcc3f772a1f85aa11334aaba82b4b821))
* **historical:** receipt a workspace-read failure in the remote worker ([ef38fdf](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ef38fdf653a815baa7aa74d153daded4ae6f3bd4))
* **historical:** receipt a workspace-read failure in the remote worker ([a0ed91e](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/a0ed91e73a5e6059edc19091942dffcda7070f95))
* **hooks:** let the commit-msg stage see COMMIT_EDITMSG (FOR-1633) ([#333](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/333)) ([fc31214](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/fc3121485e92b965bb9d5277dce78a37983d6b4f))
* **run:** publication block and verify-read retry on the final leg ([91d9711](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/91d9711dc92d5c1bf3522fc67c75b1d1cb5c4bdd))
* **run:** publication block and verify-read retry on the final leg ([#313](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/313)) ([eeef0a1](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/eeef0a1bae33fa146b40666f4645c237fe958f53))
* **run:** publish only new snapshot keys; bounded verify-read retry ([32e65cd](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/32e65cdce610ba2b748a4c5c2ac4063fe1b51d1f))
* **run:** publish only new snapshot keys; bounded verify-read retry ([#310](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/310)) ([72d56a5](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/72d56a570082d9aaaa6511d316db171cfeb9ab92))
* **run:** retain both object-transport digest sites at parent form ([9c5493f](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/9c5493fc0cc1eab138f4ecdee6f44bef077a2a32))
* **run:** retain both object-transport digest sites at parent form ([df98c8a](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/df98c8ac296ab397a169b1ccef7f5a626742191f))
* **verdict:** make the D1 vocabulary guard nominal (FOR-1573) ([0f036df](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/0f036df78365c56061da4f42685f9029411988f1))


### Code Refactoring

* **analyze:** decoders adopt strict_loads, helpers share the door ([b7634a0](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/b7634a0f3ac655e52d3e95be4e558b2eba3fa00f))
* **analyze:** decoders adopt strict_loads, helpers share the door ([8b2bf88](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/8b2bf881ef6adbe916b8c041dcf212101e21db3f))
* **analyze:** lowercase-strict SHA at every boundary site ([b7634a0](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/b7634a0f3ac655e52d3e95be4e558b2eba3fa00f))
* **analyze:** lowercase-strict SHA at every boundary site ([01f53ab](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/01f53ab7b1c996f93f9d43142b8f727fc514a620))
* **analyze:** retire the ACL output lease (FOR-1605) ([92c4d14](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/92c4d142ae7d025fff6a0a18b138ceb1b66332a4))
* **analyze:** retire the ACL output lease (FOR-1605) ([484fdfd](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/484fdfd3acd429fde6fbaf851ce7922494b77c85))
* **campaign:** share the finalization custody decision (FOR-1587) ([9fcca61](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/9fcca61436ce0462d4d69ffe0632e9c5753334bc))
* **campaign:** share the finalization custody decision (FOR-1587) ([ab1517b](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ab1517bae7158d05435375cff861a36357ed027f))
* **campaign:** type the D1 billing finalizer request (FOR-1576) ([9fcca61](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/9fcca61436ce0462d4d69ffe0632e9c5753334bc))
* **campaign:** type the D1 billing finalizer request (FOR-1576) ([86ff1b9](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/86ff1b9aab0856891c75147f4e066c6e2c68a3cd))
* **campaign:** type the D1 billing finalizer request (FOR-1576) ([#321](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/321)) ([9fcca61](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/9fcca61436ce0462d4d69ffe0632e9c5753334bc))
* **capture:** adopt strict_json in kernel and capture codecs ([4436eba](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/4436ebaae4398f214d45c3d3764efada02190676))
* **capture:** adopt strict_json in kernel and capture codecs ([#326](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/326)) ([f20025c](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/f20025c9a9660ffa3e1e26b4d50069a3d45673ba))
* **capture:** split MetricExtractor by input and narrow signatures ([d9d6759](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/d9d675993e0c5b8daf19d12ca2ac56dc82cde9ce))
* **cli:** fold d/h dispatch into the handlers table (FOR-1571) ([3770de9](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/3770de941f6eb24266f52fdc33f8a4226bd80f25))
* **cli:** fold d/h dispatch into the handlers table (FOR-1571) ([474337e](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/474337e1ae3625f65f69d34afd3ed458c963a608))
* **cli:** fold d/h dispatch into the handlers table (FOR-1571) ([#318](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/318)) ([3770de9](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/3770de941f6eb24266f52fdc33f8a4226bd80f25))
* **composition:** reuse canonical_json for D1 dumps (FOR-1582) ([3770de9](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/3770de941f6eb24266f52fdc33f8a4226bd80f25))
* **composition:** reuse canonical_json for D1 dumps (FOR-1582) ([ffb8181](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ffb8181438f4e481ed33538c5af14d3e5c6b579a))
* **composition:** type D1 runtime and RunPod builders (FOR-1583) ([3770de9](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/3770de941f6eb24266f52fdc33f8a4226bd80f25))
* **composition:** type D1 runtime and RunPod builders (FOR-1583) ([4e3c846](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/4e3c8464ae07c39823bed67a573023e82b4fe0a2))
* **composition:** type the mailbox worker_factory (FOR-1607) ([3770de9](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/3770de941f6eb24266f52fdc33f8a4226bd80f25))
* **composition:** type the mailbox worker_factory (FOR-1607) ([e3029e8](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/e3029e8306e7cbb8b0f76afd0dcdd9b9111366a3))
* **config:** own betas in TrainConfig; validate D1 via binding ([3992f9e](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/3992f9ebca37f52dc8e2171a3c87aa64f1b11194))
* **config:** own betas in TrainConfig; validate D1 via binding ([#317](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/317)) ([9377f57](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/9377f571aafeac133cadf7972d5f5eb7e28fb8c5))
* **corpus:** one offset sampler, one device dispatch (FOR-1592) ([616ab6f](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/616ab6fc4ccfe03ab5edb1c359021a0fe924b05e))
* **corpus:** one offset sampler, one device dispatch (FOR-1592) ([fc7c8e4](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/fc7c8e4b8b160db75a55288e63ebe1946286b53f))
* **evaluation:** share state_moments accessors by mixin (FOR-1611) ([5b42dd4](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/5b42dd45052f2b48d1a075f6bb32cb1ba869f2dd))
* **evaluation:** share state_moments accessors by mixin (FOR-1611) ([2e7c819](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/2e7c8198ec543f750c7ff389ef639f29fc1e7c1b))
* **gate:** delete the dead D1 family adapter (FOR-1574) ([92c4d14](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/92c4d142ae7d025fff6a0a18b138ceb1b66332a4))
* **gate:** delete the dead D1 family adapter (FOR-1574) ([df26f92](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/df26f92067f56fc79c34a4f92261c2134756fe13))
* **gate:** delete the dead D1 family adapter (FOR-1574) ([#325](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/325)) ([92c4d14](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/92c4d142ae7d025fff6a0a18b138ceb1b66332a4))
* **historical:** add a local resource-phase helper (FOR-1586) ([ef38fdf](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ef38fdf653a815baa7aa74d153daded4ae6f3bd4))
* **historical:** add a local resource-phase helper (FOR-1586) ([6745019](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/67450194d8f01c2ad63652ec99c9dbf202782584))
* **historical:** delete load_ladder_model (FOR-1606) ([ef38fdf](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ef38fdf653a815baa7aa74d153daded4ae6f3bd4))
* **historical:** delete load_ladder_model (FOR-1606) ([c267d8b](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/c267d8b8bc07f3f34d12eb2d5d760a9bae5f2ade))
* **historical:** lowercase SHA at the remote boundaries ([ef38fdf](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ef38fdf653a815baa7aa74d153daded4ae6f3bd4))
* **historical:** lowercase SHA at the remote boundaries ([84594a8](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/84594a892d14a120e2180a4a4f32f35fa07091c0))
* **historical:** reuse require_lowercase_digest at the boundaries ([ef38fdf](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ef38fdf653a815baa7aa74d153daded4ae6f3bd4))
* **historical:** reuse require_lowercase_digest at the boundaries ([17ea424](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/17ea424b2e230ce6f5f0db89fea6c99c8f5f47cf))
* **historical:** share optimizer and autocast helpers (FOR-1570) ([ef38fdf](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ef38fdf653a815baa7aa74d153daded4ae6f3bd4))
* **historical:** share optimizer and autocast helpers (FOR-1570) ([e6d14c7](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/e6d14c7a13f84c1e83b9bf6d644f3dd7800ef12e))
* **historical:** share optimizer and autocast helpers (FOR-1570) ([#322](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/322)) ([ef38fdf](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ef38fdf653a815baa7aa74d153daded4ae6f3bd4))
* **identity:** delete the legacy EncoderShape contract (FOR-1590) ([92c4d14](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/92c4d142ae7d025fff6a0a18b138ceb1b66332a4))
* **identity:** delete the legacy EncoderShape contract (FOR-1590) ([63bbccb](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/63bbccbf12a229a10372de82ce077ca5d2b2024d))
* **identity:** land the p14 identity rail and re-pin readers ([835662e](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/835662e92201f927ded98c736163d1313696c284))
* **identity:** land the p14 identity rail and re-pin readers ([#331](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/331)) ([21a30f6](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/21a30f6d659abe075aa4a1ec68db69320d95cb28))
* **identity:** rename require_sha256, land caller audit ([69a17fc](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/69a17fc4b311d4a5bc72e26b573e92f6ef95ccda))
* **identity:** rename require_sha256, land caller audit ([#315](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/315)) ([0cee7cd](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/0cee7cdfea62a101d63733507246c2eac2f202c2))
* **identity:** retire require_sha256_hex_any_case (FOR-1636) ([#336](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/336)) ([ecc6e88](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ecc6e88e945d870e6fbb848a0e5947ba60aa9014))
* land p14 consolidation sprint ([3ca01bd](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/3ca01bd79594821aea87ae8979831a86864a6de0))
* **latent:** delete the targets re-export facade (FOR-1594) ([92c4d14](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/92c4d142ae7d025fff6a0a18b138ceb1b66332a4))
* **latent:** delete the targets re-export facade (FOR-1594) ([451b202](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/451b202aa4c949b9024647d0044d4a3a37c2085b))
* **latent:** one typed accessor for horizon heads (FOR-1595) ([616ab6f](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/616ab6fc4ccfe03ab5edb1c359021a0fe924b05e))
* **latent:** one typed accessor for horizon heads (FOR-1595) ([f7da1cb](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/f7da1cbdf3465acf962b4df68c3af9444929a6f2))
* **observation:** delete unused byte codec aliases (FOR-1593) ([616ab6f](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/616ab6fc4ccfe03ab5edb1c359021a0fe924b05e))
* **observation:** delete unused byte codec aliases (FOR-1593) ([af83723](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/af83723d9e24e96e78b40ebd0690ee96ec63b9cc))
* **observation:** delete unused byte codec aliases (FOR-1593) ([#324](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/324)) ([616ab6f](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/616ab6fc4ccfe03ab5edb1c359021a0fe924b05e))
* **observation:** share one decode validator (FOR-1591) ([616ab6f](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/616ab6fc4ccfe03ab5edb1c359021a0fe924b05e))
* **observation:** share one decode validator (FOR-1591) ([b38976e](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/b38976eb2fcfb42025892008ff564ae673d37690))
* **register:** adopt strict_json in register and gate codecs ([388ed49](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/388ed49c1f0c4bb99633380f0333a6642d0e4180))
* **register:** adopt strict_json in register and gate codecs ([2a958d2](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/2a958d2a6a383c9a948a28d268ee9f213c6da8c5))
* **register:** adopt strict_json in register and gate codecs ([#319](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/319)) ([388ed49](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/388ed49c1f0c4bb99633380f0333a6642d0e4180))
* **register:** delete d1_configs_dir alias (FOR-1600) ([92c4d14](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/92c4d142ae7d025fff6a0a18b138ceb1b66332a4))
* **register:** delete d1_configs_dir alias (FOR-1600) ([df018fc](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/df018fce0df2bcd45b07861c43fe93796d825d14))
* **resource:** type meter restore and its activation calls ([9fcca61](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/9fcca61436ce0462d4d69ffe0632e9c5753334bc))
* **resource:** type meter restore and its activation calls ([d51f8f5](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/d51f8f5ef16363dc3bede8fd52b3c01092573134))
* **run:** adopt strict_json in ladder/run and the D1 CLI ([9c5493f](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/9c5493fc0cc1eab138f4ecdee6f44bef077a2a32))
* **run:** adopt strict_json in ladder/run and the D1 CLI ([eee51fa](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/eee51fa8af8757e71a42ef3d79a5c1fe56c1dcb0))
* **run:** adopt strict_json in ladder/run and the D1 CLI ([#328](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/328)) ([9c5493f](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/9c5493fc0cc1eab138f4ecdee6f44bef077a2a32))
* **substrate:** move GatedKappaNorm beside build_norm (FOR-1597) ([616ab6f](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/616ab6fc4ccfe03ab5edb1c359021a0fe924b05e))
* **substrate:** move GatedKappaNorm beside build_norm (FOR-1597) ([3648a00](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/3648a001a270ac50da4e62389d0db58f1e85cb31))
* **trainer:** one failure helper for fresh and resume (FOR-1568) ([5b42dd4](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/5b42dd45052f2b48d1a075f6bb32cb1ba869f2dd))
* **trainer:** one failure helper for fresh and resume (FOR-1568) ([559fa51](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/559fa5125cabf0707c2e63a4d822151da15ea04c))
* **trainer:** one failure helper for fresh and resume (FOR-1568) ([#320](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/320)) ([5b42dd4](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/5b42dd45052f2b48d1a075f6bb32cb1ba869f2dd))
* **verdict:** hoist review status sets to constants (FOR-1573) ([8141be3](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/8141be30bb97d62ef74e5b0da02d61e26dd6a2f0))


### Documentation

* **plan:** add the p14 consolidation plan ([d87ac4b](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/d87ac4b73f300ce90c1a49cc640d7cc28d784b68))
* **refactor:** file the thermo-nuclear review at the p14 epoch ([ba109b8](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/ba109b8492957f694142f4fae493a896cf185af7))
* **sprint:** add the fifteen p14 phase briefs ([5e24085](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/5e240850851f13b24dc2001fb59a8ac5090b055e))
* **sprint:** add the four p14 pre-phase rung briefs ([cdb99aa](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/cdb99aa481e1daad109a461c0a4f50b58d82b3c5))
* **sprint:** add the pre-for-1567c rung brief (final-leg publication) ([3c8db2a](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/3c8db2a3578d43ae33405efb9543a3ccd7e30c04))
* **sprint:** add the pre-for-1622 brief and plan rungs rows 6 and 7 ([9d90e4c](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/9d90e4c421b96899c67ded8679ae8255f21526bd))
* **sprint:** order w1/w2 members by measured footprint ([4e0ae6a](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/4e0ae6a59117d054cc9b63d705d5004e06b982cb))
* **sprint:** rule the pre-for-1567 design-gate decisions ([9056936](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/90569362cfe5d62dca81b1e978fc0cc4fff87fdf))
* **sweep:** p14 prose fixes, FOR-1626 annotations, register rows ([29cc380](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/29cc3806859a8ca18331cc5d13b29705c7a1386e))
* **sweep:** p14 prose fixes, FOR-1626 annotations, register rows ([#332](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/332)) ([da4d35c](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/da4d35cbff05aa61b234b1a1e080e24303225eed))
* **sweep:** wave-1 prose fixes, audit annotations, two retractions ([b11855d](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/b11855db5decdc0bf4eec64dcd71e976b18bebb7))
* **tests:** rename require_sha256 mention in a launch-attempt docstring ([d7989af](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/d7989afdb2df2646e8fdd531b390ed0a441056de))
* **verdict:** audit persisted D1 verdict vocabularies (FOR-1573) ([a322ad1](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/a322ad168db155c8fca1643aaaa7b23c6c8afe61))
* **verdict:** audit persisted D1 verdict vocabularies (FOR-1573) ([#330](https://github.com/Luiz-Frias/entropic-substrate-lm/issues/330)) ([07048f0](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/07048f05991fe3b68f22a7e865d05184a11f4d27))

## [0.1.0](https://github.com/Luiz-Frias/entropic-substrate-lm/compare/v0.1.0...v0.1.0) (2026-09-28)


### Bug Fixes

* **tests:** run marimo check via sys.executable, never a re-syncing uv run ([6432ea9](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/6432ea9af1bff96b72cf833179e11ee7c488e18f))


### Miscellaneous Chores

* **release:** bootstrap release-please at 0.1.0 ([d827419](https://github.com/Luiz-Frias/entropic-substrate-lm/commit/d827419509dfadb3f2186d8271a81649a169c29e))
