# PLAN

## v3.2 — diagnose the zero-model regression

Active goal follows the Pro review of 7699299. See
`docs/pro_review_7699299.md`, `docs/v3_2_protocol.md` and the evidence
ledger in `docs/v3_2_requirements_result.md`.

- Frozen score path, absolute classification tables and reliability complete
  in `docs/v3_2_opened_result.md`, including the frozen QRF distance comparison.
- Repeat `python3 scripts/v3_2_preservation_check.py` before final delivery.
  The current audit verifies 8,773 protected paths against 7699299 in the
  commit, index and worktree. Check data source digests with the run receipts.
- Fit the population by SED tilt comparison with matched selection. The
  full set of 32 initial shards is complete and verified.
  The compact receipt audit independently counts 26000 broad PSFs and
  3540 candidate PSFs.
  Remaining input shards checkpoint completed row integrals and reuse only
  authenticated rows. Assembly checks the archived shard 00 receipt.
  Independent complete audit verifies all 731 reviewed contexts pass.
  Fitting verifies each check's identity, admission and quadrature recipe.
  Run `PYTHONPATH=src:scripts:.deps python3 scripts/v3_2_tilt_input_audit.py --complete`
  before final input sign-off.
  Array 61259727 uses aftercorr on
  61259514 so each completed input shard can enter refinement. It
  checkpoints the exact flagged (row, identity, admission) contexts.
  Raised the input array throttle from 4 to 5 after all eight pending
  slots reported JobArrayTaskLimit with no running slots. The scheduler
  cleared that reason and retained both downstream dependencies.
  Shards 24 and 25 moved to debug QoS with their existing 30-minute limits,
  within the two-running-job QoS cap. Refinement dependencies still apply.
  Four matched fits (61259811), predictions (61259969), and the paired
  report (61259977) are queued in dependency order. The fit also records
  fitted-draw entry transfer for both alternative quadrature levels.
  The paired report tracks their distance-bin counts and external scores,
  and decomposes the historical regression with an exact same-object sum.
  Numerical summaries retain flags for each available score channel.
- Decompose giant branch availability, suppression and class-conditional
  distance error using the registered strata. Frozen class probe job
  61260227 completed on 519 opened PSFs. See
  `docs/v3_2_opened_probe_and_count_result.md`. Choose one supported
  modification after the matched factorial results.
  The factorial report includes same-object class-score contrasts with
  the four frozen photometric probes.
- Run scoped narrow-structure and numerical diagnostics. Registered 288
  catalogs in `docs/v3_2_radial_protocol.md`; design job 61260010 complete.
  The atlas shows all 288 realized after-selection member fractions.
  All three original direct pilots in 61260177 timed out. Direct 000
  follows as 61260934; empirical 128 (61260935) and prior-predictive
  256 (61261884) use completed repeat-00 cache 61260465.
  Catalog 128 completed with both numerical verdicts passing. Its true
  width of 0.08 mag falls outside the local 95% interval [0.096, 0.998] mag.
  Direct 000 completed in 29m35s. The local sampler fails Rhat and ESS
  despite passing all entry and count checks. Keep the first matched
  model/empirical comparison in the fitted subset.
  Prior-predictive 256 completed. Its local sampler also fails Rhat
  and ESS, with all entry and count checks passing.
  Exact direct/cache verification 61261765 completed and passed on both
  pilot arrays of 128 contexts. Cache array 61266329 (repeats 01–31,
  max four concurrent) has cleared that dependency. Cached recovery 61266369
  covers the 285 nonpilot catalogs with at most eight concurrent tasks,
  waiting for the cache array and prior-predictive pilot 61261884.
  Report 61266374 waits for the recovery array and both outstanding pilots.
  Workers retain authenticated completed rows across allocation limits.
  Cache receipts 00–04 are verified.
  Final reporting pairs broad and local count error by catalog and
  retains missing or excluded catalogs in the coverage bounds.
- Prepare the DR19 Astra identity inventory following
  `docs/v3_2_reference_intake.md`. Archive acquisition 61260678 completed.
  Initial index 61260891 exposed a release-column mismatch. The corrected
  census 61264018 recovered 39,505 exact-identity requests. Native match
  61260892 found 100 exact primary rows and admitted zero because all
  failed photometric entry. Job 61267191 verified that all 100 fail the
  grz mask, grz fraction quality and low-reddening conditions. Assess a
  separate outcome-closed acquisition route for the same science domain.
  The empty DR19 comparator 61261381 was canceled before execution.
  DR20 BOSS MINESweeper acquisition 61267444 completed. The LCO identity
  census and source limitations are in `docs/v3_2_minesweeper_intake.md`.
  Census 61267684 completed with 7,297 native match requests. The original
  PSF and photometric admission runs on them in job 61267848.
  Freeze model, rules and new reference roles before
  scoring the new validation outcomes. DR19 receipt/evidence scripts bind
  all four complete forecasts and verify the exact reference row pointers.
  The paired evaluator covers signed Gaia, class and teacher distance/entry.
  Reliability bins and expected versus observed counts use the same
  qualified reference rows in each view.
  The frozen nominal, historical target and QRF forecasts for DR20
  follow nonempty native admission and freeze the exact photometric candidate gate
  for all native PSFs. Report paired and absolute metrics inside the
  candidate gate and across the complete native reference cohort.
  Candidate forecast awaits the mechanism decision.
  After the forecasts are assembled, the final evaluation Slurm entry
  point runs freeze, evidence access and scoring in that order.
  The opened-label coverage and requirements for a parent-matched
  contamination measurement are in `docs/v3_2_contamination_support.md`.
- Inventory new common-distance HST methods. Job 61262297 admits 77 PSFs
  in 47 Tuc and 26 in NGC362. Frozen comparator job 61264223 uses the
  same photometric gate as the field cohort. The allocated final
  runner binds all four forecasts before evaluating both HST scales.
  Keep the existing v3.1 system as an opened control.
- Trace the 33 frozen conditional count failures with a statewise weight
  versus response audit (61260338 complete, all 33 weight-only at the
  worst bin). NGC288 odds test 61259650 completed at fixed distance order
  64. Both branches pass 512 to 1024 and retain failures at 256 to 512.
  Added-distance pilot 61260630 passed all 15 checks, with exact native
  closure at the old nodes. Full 32-state integration uses the registered
  levels with per-state checkpoints (61263907). The two-axis evaluator
  61263908 waits on all 32 states.

## v3.1 — completed research delivery

Current objective: preserve the completed v3.1 deliverable; final source-bound
delivery gate passed. The final commit is synchronized before goal closure. All authorized science
computations, nine result reports and22visually reviewed plates are complete.
Scientific rejection remains: no adopted correction/calibrated domain, no measured
independent fullcandidate contamination, no new independent NGC288 scale recovery.

### Current execution snapshot

- All102transfer states,64response probes,176radial catalogs/352arms and once-opened
  external validation complete. Finalscore audit432132checks;183tests pass.
- All32structure states assembled oncompute; originalnative490/entry11052flags
  retained,0unresolvedpreparatorycontexts. Both joint quadratures fail; both65536
  shared-countdraws pass but cannot override posterior rejection. Finaldomain
  join3540PSF/fixed3003highq/75cells/30occupied establishes0calibrated cells.
- All22English plates visually reviewed and319unifiedfigure checks pass.
- All21requirements and current narrative authenticated; actual full delivery
  gate passed. Finalcommit/sync is checked after freezing this record, before
  closing the goal. No sciencejob remains. Do not submit newmodel/refinement/
  catalog/deployment/timer/agents.
- Future science requires a newly authorized frozen scope. Evidence gaps and
  failure causes are described in docs/v3_1_result.md, not deferred as hidden jobs.

The historical execution notes below are preserved for recovery; this snapshot
supersedes their pending-job and partial-delivery statements.

### Milestones and current evidence
- Resource execution: all4exactresponse groups61229862 completed; full64state
  report61221949completed6m49; fullresponse scientificresult/2plates accepted
  asartifacts, notscientificcalibration (newESS1.023628, original/pooledfailed). Structure-native replacement61231239 (max7pairs, unchanged
  32workers,max2nodes) submittedafter56zero-runtime cancellationproofs;8prior
  completehashes retained. Twoactual-product structure figure scripts prepared;
  full32state application/render/visualreview stillrequired.
- [x] Read latest full Pro review/related chats (noDOCX), register goal/roles/ledger.
- [x] WP1 implementation and scientific interpretation complete; correction NOT
  accepted. Source-backed docs/v3_1_mechanism_result.md,76checks/28sources;
  all11882stratified contrasts retained. Same1542development stars/17386views and
  pairedclass/FeH/band/prior contrasts complete. Response12 targetprior worsens
  evolved reference performance. Six-parameter luminosity correction has32shared
  sky bootstraps;96numerical/15heldout guards pass but evolved scatter is weakly
  constrained. Full latent2589views plus failure pairing complete: low-negative
  evolved297primary references still87.6635%severe |dmu|>1; negative-flux outliers
  alone do not explain giant failure. Rare unknown/W2 support flags retained.
- [x] WP2 target/systematic comparisons complete; full radial experiment/report
  complete. All176catalogs/352arms retained: broad163/local105pass,97bothpass.
  Originaltimeouts andterminalfailures retained. Correctedreport61229960
  COMPLETED45s; exact SharedCounts normalization comparison replaces raw-weight
  comparison only (old61228930FAILED22s retained). Correlatedsummary61228931
  COMPLETED16s/29304rows. All21empirical-source scenarios collapse toone source
  block; no source interval. ThreeEnglishplates/10panels visuallychecked,
  1823artifactchecks; fullscience/limitations in docs/v3_1_radial_result.md.
  Narrow4kpc/.08all8numericalpass butlocation3/8,width0/8coverage despite
  countL1 38.08%→24.66%; notscientificacceptance.
  Native target26000rows across34corrections complete,258fineaudits allpass.
  Direct entry all16shards complete:26000objects/29540envelopes allpass;
  initial61222996/61222997 resourcegroups bothTIMEOUT; exactresumes61223947
  (6–9)/61223978(10–13), unstarted14–15group replaced61223886. Alloriginals
  preserved. Assembly61219399 COMPLETED28m10/allpass; jointrepack61222999 passes.
  Candidatezero61224010 COMPLETED1m49:512countMCfailure retained,
  2048retained passesallsampler/bridge/count/numericalgates. Remaining101states
  running/queued61225864; skipbatchindex34 preservespilot. Ensemblejobs broad
  61225883/candidate61225884/holdout61225886 waitforexactgroups0–8/8–16/17–25.
  All34candidate states pass2048; ensemble61225884 completes18s. MainESS21.17325,
  maxstateweight.07015244; acceptedFalse because seed absolute logZdelta.248538
  exceeds.05. Probability/count/weight sensitivity gates pass; do not replace
  mainwithcut orrelax evidence gate. All34broad conditionalstates nowpass:
  33at2048 andstate23at8192. Original512/2048failures retained; group61225864_5
  TIMEOUT30m12 duringjointcounts preserved. Exactcount-onlyresume61232648
  COMPLETED25m13, allthreecachedfits reused andfinal8192sampler/bridge/countpass.
  Broadensemble61225883 COMPLETED24s/authenticatesall34payloads; mainESS30.97863
  andmaxweight.0379785 butabsolute logZseed+.600975/previouslevel−6.637352 fail
  .05gate. Zero/point/main rejected; cutpasses butnotreplacement. 229actual
  compactchecks pass; candidatecompact also229checks passafterretrieving25
  missinglocalmarkers againstensemblehashes (no rerun). Floatingweight/count roundoff andinitialcheckerfailure
  preserved separately, productiondata/scientificgates unchanged.
  All34 holdout statescomplete:33at2048/state1at8192; originalfailures retained.
  Lastgroup25 COMPLETED8m00; ensemble61225886 COMPLETED25s. MainESS30.91517,
  maxweight.0462794; absolute logZseed+.464854/level−5.016593 fail.05gate.
  All229compactchecks pass, cutnotreplacement. All102conditionalpopulation
  fits/threeensembles complete; full102state predictive transfer remains.
  Fullensemble andfixed3003-ID highq61226224 complete6m17, inheritedmainflag;
  expectedfar8.00304=.266502%, notmeasuredcontamination; MCprecision retained.
  Heldoutfields12/4/6 frozen;
  Pendingtransfergroups5/17 containing8192states now2h (4x17–18min estimate
  exceedsold1h). Guarded12-state2048resourcepilot61237364 COMPLETED18m39:
  all12states/96conditionalchecks pass; singlechild1.485GiB/batch19.42GiB,
  all57originalsources rechecked. Pilotzero-runtimecancelsonlygroups3/4/6.
  Measuredsameexecutor rollout61238249array0–4%2 nowqueuedfor60states after220
  sourcehashes and15zero-runtime cancellations (old7–16/19–23). Report61231678
  nowwaitsretainedoriginal5/17/24/25 +replacementarray+holdoutensemble, no
  canceledparent/children; max2packednodes, originalarraythrottle2 retained.
  Firstrollout61238249_0(raw61238416) COMPLETED18m31:12executionsverified,
  90/96numericchecks pass;6candidate0–5 broad-entry_level countchanges
  .60561–.66554>.5 retained. Broad28–33pass. Heldout28–31also COMPLETE17m34/32numericchecks pass. Secondpack61238249_1 COMPLETE17m41, candidate6–17eachretainonecountgatefailure;
  holdout32–33 COMPLETE17m21/16checks pass. Thirdpack61238249_2 COMPLETE18m45/85of96checks pass, candidate20passes
  all; other11retainbroadentry_levelcountfailures. Total70/102transferoutcomes,
  1401reproduciblecompactchecks pass,29originalnumericfailures retained.
  Firstsixcandidate in-sampleunitjointESS1–1.015broad versus848.225–996.181
  candidate;12/24basejointsupportdiagnosticspass, notindependentvalidation.
  Existinginitialentrypilot61239060 promoteddebug30 after101sourcehashes and
  >18actualsingle-state timings; combinedruntime unmeasured/no wideradoption.
  Existingresume61239061 promoteddebug30after414loggedstructurecompletions
  andfreshcap; actualcacheverification oncompute, no duplicatewriter.
  Lasttwoholdoutstates task25 andwholeholdoutensemble arealreadycomplete.
  Firsttrueholdoutgroup18 completed17m56/states4–7,32numericchecks pass butall
  16basejoint supportgatesfailESS; unitESS5.147–10.017(broad)/2.458–10.558(candidate).
  Populationfitexcludesfields4/6/12; complete102state mixturestillrequired.
  Entry12/13completed13m21/13m07 with12/512originalflags retained.
  Initialentrypilot61239060 nowqueuedfor30/31 (same32workersperstate);
  101smallsourcehashes authenticated,4payloadhashes checkedoncompute. First
  proposed13/14abort leftthosewriters intact;30/31 bothcancelledzero-runtime.
  Controller31expiry recoveredvia fullparentarrayaccounting. Originalentry
  arraythrottle is4; originaltransferarraythrottle2. Actualpilotmeasurement
  stillrequired beforewiderpacking. Entry9original512failures retained; existing
  pendingentry10/11/refinement2 promoteddebug afterfreshcap check.
  shared whole-field predictive/count transfer andreport drivers nowready, with
  proper joint-state integration and explicit out-of-sample/support flags.
  Independently regenerated32SED refits nowcomplete: exactgroup7resume
  61227619_7 COMPLETED14m07, allfourconverged at25/26/21/18iterations. Original
  timeout/checkpoints retained; all64probe/refinement/report outcomes nowcomplete.
  New32numericpass butESS1.023628; original/pooled blockedbyone response9target
  nativefailure. Fullresponse result and2visuallyacceptedplates delivered.
  Matched64-response fixed-prior probes frozen128target/256seen-development,
  pilot old12/new1004 completed61220906 and both pass; remaining62probes submitted
  with individual refit dependencies. Development numericalsidecar driver now
  preserves originalfailures andrequiresall64inventories before finalreport;
  response0all3originalfailures pass.005sidecars, all256objects retained.
  These are not26kpopulation-marginal evidence.
- [x] WP3 frozen final evaluation and evidence-gap audit complete; scientific
  validation fails, no calibrated-distance claim. Roles frozen:170strictnewsky+356source-new
  seen-sky field and3032NGC1261common-scale PSF objects. Only9precisedistances in
  strictnewsky; no new far-field precision evidence. All3558reference-prior
  numerical sidecars pass:field526at.01;common2998at.01/26at.005/8at.0025.
  Originalflags retained. Allpredictionfreeze61226412 COMPLETED5m33/53725artifacts; fieldreference
  association61226762 openedreference at14:50:09UTC afterimmutablefreeze;
  associationcomplete15m37:494signedGaia/40negative,126wholecohortprecise
  distances;0MWSmatches. Final target-prior
  identity computed withretained numericalflag. Both8-ID targetprediction pilots
  numericalpass; fullfield61226370 completes4m28, all526numericalpass; fullcommon
  61226371 completes23m53, all3032numericalpass; originalensembleflagretained.
  Sharedfield checkpointreuse verified. Oldtargetfield61222759 completes526objects/allpass;
  common61224421 completes22m43, all3032pass. Finalfreeze/evidence/nine-view score drivers andprotocol
  nowimplemented with exactidentity/hash closure and actual saved-resolution tail
  checks. Dependency-gated evidence61226762/64,score61226763/70,report61226775,
  domainavailability61226776/relativescreens61226777 submitted; dependentexecutionpending. Shared-scale/sky aggregate report nowimplemented; strict
  candidate admission/primary versusoutside-domain report protocol added, exact
  application3540mask reproduced by61224943(20s). Field519/526 andcommon1732/3032
  in samecandidate domain; others retained outside-domain. Same-ID MWS/VAC channel
  association/scoring implemented; bothassociationscomplete (common2419signedGaia/996negative; oneHSTscale).
  Fieldscore61226763 complete4m37:all4734scorenumerics pass,1578inheritedflags; commonscore61226770complete5m00:27288rows/72scorenumericalfailures
  on38objects, allretained; mainperobjectscorenumerics pass. Report61226775
  COMPLETED6m05; bothcohorts/nineviews retained. Newsky Gaia main-oldtarget
  -0.8143nat[ -0.9342,-0.6137 ]on166stars/9cells; NGC1261candidate1732median
  distancebias-49.05%, onecommon scale. Exhaustiveaudit61228029passes432132
  artifactchecks; all72newscorefailuresareGaia. No scientificacceptance. Fullsource-qualified
  report v3_1_final_validation_result.md documentszero5–10kpcnewskyreferences,
  no precise distancesfor77poornewskygiants, andhighqreference-selection gaps.
- [ ] WP4 product-matched evaluation/high-q counts/domain/structure example.
  Exact main candidate mode decomposition is complete: all 8 shards / 3540
  objects / fixed 3003 high-q; report 61227828 completed 6m15. All per-object
  gates pass, inherited main evidence failure remains. Multimodal 2241 / 2013;
  overlapping class/binary/parent responsibilities are not physical labels.
  One English plate / four panels inspected; 3613 actual checks pass. Original
  pilot roundoff failure and exact retry preserved; full results and modes stay
  source-linked in docs/v3_1_candidate_modes_result.md.
  True-distance Gaussian correction integral and shared count/numerical-transfer
  foundations implemented/tested. Joint response61220368 all26000rows pass.
  SeenNGC288512photo-only reference-ensemble structure protocol registered,
  analytical baseline withdrawal/entry/math foundation tested; preparation61221747
  complete61221747. Nativepilot7/512initialfailures allpass.005sidecars61222475;
  all32native/refinement states nowcomplete, originalfailures retained. Entrypilot all6branch-context gates
  pass aftersame-integrand2^23extension61222758. Originalfailure preserved.
  All32state entry61223167/refinement61223172 underway; queuedrefinementindices
  extendedto3h. Originalrefinement0 TIMEOUT30m27; terminal/absence/sourcehashverified,
  unstartedresume61225943replaced byresource-only32workerpilot61228245debug30min
  afterterminal/absence/hashchecks. PilotCOMPLETED26m06:all512contexts/1024
  branchrows pass,original512failures retained; maxchildRSS3.967GiB. Prepare
  same32workerexecution nowarray61229422for30states2–31 aftercorr61223167;
  originalstate1nowcomplete1h58m16/all512pass; refined3complete2m28/all512pass
  with24originalfailures retained; refined4complete8m54/all512pass with124originalflags; refined7complete6m35
  with81originalflags,0unresolved. Pendingstates2–31 now1h
  (>2x exact32-worker pilot), max3nodes; native/refinedall32complete,490original
  nativefailures retained and32/32refined markerpass. Fullpayloadcheckinassembly.
  Complete-state chain submitted: assemble61232760 -> infer61232761 ->
  report61232762. Assembly nowrequires explicitunfinishedentry14–29, pilot61239060,
  refinement2resume61239061 andallunfinishedrefinement5/6/8–31 children;
  completednative/entry/refinedstates authenticated. Canceledparent andtimedout
  child excludedonlyafter source/accountingproof. Refinement2TIMEOUT30m04
  retained; sameadapter resumesonlyunfinished per-context/per-seed checkpoints
  under1h allocation. Ref30/31 dependonpilot beforeassembly. Allstates/numerical gates mandatory. Pendingentry1–31limits now30min basedon
  13m37pilot; pendingrefinements now1h (>2x exact32-worker pilot). Pending
  populationgroups restored2h afterbroad23timeout; originalcheckpoints/seeds/scientificschedules unchanged. Separate shared realizedcount/seen-Gaia
  comparison report nowimplemented/tested, notrun; originalpriors/likelihoods unchanged.
  Photometry-only75-cell grid andpost-freeze domain evidence/support drivers now
  implemented/tested; photometry-onlypreparation61225647 completes8s/allPSF,
  no outcomes read. Post-freeze evidence 61226776 completed 5m03: 30 occupied
  candidate cells; 2810/3540 candidates lack precise new-sky references in their
  cell. Mean unsupported model distance mass 79.5834%; complement not validated.
  Two English plates / 12 panels inspected; 1309 actual checks. All 600 joint
  count groups conserve objects; nine precise new-sky AstroNN references occupy
  five photometric cells. Source dependence and inherited main failure remain.
  Relative screens complete; final scientific adjudication requires remaining
  population/transfer and structure results. See v3_1_domain_evidence_result.md.
- [ ] Final reproducible reports/ledger/README/AGENTS/visual QA and safe sync.
  Integrated methods nowin docs/v3_1_methods.md, source-linked to protocols, status, ledger and the completed external report;
  equations/product identities cross-checked against code. This isnotfinalacceptance.

### Validation and next actions
- Source-level review now coversall21 ledger requirements/68evidence bindings;
  E3/E4/F1/F2/F3 remainpending. D2/E2 explicitlyretain unavailableindependent
  calibration/contamination evidence andscope limits. Finaldelivery additionally
  requiresfinal_requirements_review.json with exactcurrent IDs/text, authority
  hashes, findings/limitations andactualevidence. Partialreview cannotclosegoal.
  Syntax/currentmissing-full-review guard pass; fullsuccess stillawaitsproducts.
- Current baseline guard verifies6976 tracked baseline files in worktree AND index,
  and rejects additions outside v3.1 namespaces. Final delivery repeats it live.
  Four temporary-repository regressions pass; full183tests pass7.58s/164source
  hashes. Previous179-test evidence retained. Current missing-input guard passes;
  complete structure/domain products stillrequired for a successful final gate.
- Full102transferreport61231678 COMPLETED1m15,625actualreportchecks/96countgroups;
  allmain inheritedevidence failures and4mixed-sensitivity failures retained.
  HoldoutunitjointESS103.872broadpasses/21.669candidatefails; broad→candidate
  ESS388.104buthalfdelta.8532fails. TwoEnglishplates/12panels visuallyaccepted,
  158source/formatchecks; now19/22plates. Committedreview provenanceaudit
  nowpasses270checks over19plates; final22guard remains.
  Existingstructure24andnew25promoteddebug30 afterfreshcap; no newjobs/science.
- Finaldelivery gate scripts/v3_1_delivery.py prepared: requiresactual102transfer,
  all32structure/assembly/posterior/counts, ninewrittenreports,22visuallyreviewed
  figures, freshunitrecord andcompletehash-boundnarrativereview. Syntaxandactual
  missing-input rejection pass; no finaldelivery marker exists oracceptanceclaimed.
  Freshscientific tests106v3.1pass; completev3+v3.1suite179pass6.62s/162sourcehashes.
  Collected52missingnative/refined-native markers (now32each), PSF/state/source
  andoriginal-markerhashlinks checked; largepayloads remainallocatedassembly.
- Unifiedfigure checker scripts/v3_1_figure_delivery_check.py defaults toall22
  namedplates; latestaggregate verifies19plates/270checks. Earlier17plate/240
  snapshot andmissing5guard retained. Renderandvisuallyreview structure2/domain1,
  commitacceptance provenance, thenrunfull22check; scientificacceptance separate.
- PredictiveESS isweight concentration1/sum(w²), notautocorrelation-adjusted;
  interleaved4chain parity compares0/2versus1/3. Report/figure wording clarified,
  productionvalues/frozen thresholds unchanged. Reauthenticate19figureprovenance
  afterlabel-only visualreviewcommiteca7352:270checks pass, originalrowsunchanged.
  Finaldelivery gate additionallyrequiresall16384structurestatus rows,512sameIDs,
  fixedroworder, explicitflags andper-state original/refined marker reconciliation.
  Syntax/currentmissing-input rejection pass; fullsuccess awaitsactualmatrices.
- All102/102conditionaltransfers authenticate2042compactchecks;33original
  numericflags (candidateexcept20) retained. Fullcandidate34 conditionalbase
  support passesonlysamecandidateenvelope; all30collectedheldout conditional
  supportgatesfail. Original5COMPLETED17m34/all4pass, broad34complete; onlyholdout0–3
  fromoriginal17 werepending; nowCOMPLETED17m33/raw61235900/all4pass.
  Existingreport61231678 promoteddebug30 after108sourcehashes/zero-runtime/
  dependency-null/freshcapacityproof; no newsubmission, fullmixturenowcomplete625checks/2plates.
  Same2048/8192workload measuredparallel, sodebug30
  fororiginal17 afterpending24returnedregular30. InternalQOSregular_1failed;
  publicregularsucceeded, originalerrorpreserved. No mixtureacceptance.
  Refined11/23 complete10m36/2m06/all512pass, original134/3flags/10checks each;
  refined21also complete8m32/all512pass, original103flags/10checks retained.
  Refined9/22complete25m51/10m38, original512/133flags retained.
  Refined19complete11m01/original153flags;13/26complete21m50/10m53,
  original512/149flags. Refined14/20also complete19m17/17m06, original268/233
  flags retained. Newarray17/raw61247266 COMPLETED21m29/all512pass,512oldflags.
  Newarray18/raw61247267all512pass,397oldflags; Python1052.386s.
  Newarray15/raw61246950 COMPLETED21m24/all512pass,512oldflags; Python1273.713s.
  Total32/32refinements authenticate702groupedchecks withnewreproducible
  refinementcompactchecker;11052originalfailedcontexts/0unresolved preserved.
  Missing32stateguard passedwithoutoutput. Localmissingresourceaudits3/4/7
  retrievedagainstunchangedmarkerhashes; no scientificrerun.
- One64-worker pilot13 submitted61244889 aftersource/zero-runtimeproof,
  heldsubmission/dependencyrepair/release. 320GiBguard, unchangedoriginal
  function/seed/inputs. Pilotcompleted21m50/Python1293.38s/all512pass; batch
  199.7GiB/maxchild<4.1GiB. Preparedsameexecutorarrayforpending15–18/25/27–31,
  Array61246935nowholdsremaining15–18/25/27–31, max3newnodes plusoriginal
  sparse14/20/24<=6. All10oldwritersprovedCANCELLED/zeroelapsed/noAllocTRES
  andqueueabsent; oneheldsubmission, repairedassemblydependencies before
  release.40receiptchecks/19sourcehashes pass. Last31controllerexpiry and
  twofailedrecoveryguards preserved; no repeatcancel/duplicatewriter.
  Subsequentdebug15/16queueadjustment returned successfully, bothpromoted.
  Collectedhash-verifiedjournal/complete; earlierlivecommand not repeated.
  Disjointnewchildren17/18 promoteddebug30 underfreshcapacity, reserving space
  for15/16 andlastheldoutgroup; no duplicatewriters/newsubmissions. Lastheldout
  original17nowcomplete; originalstructure14/20complete,24retainedregular30.
  Originaltransfer5complete;17promoteddebug30afteractualmatchedworkload.
  Sparse24temporarilyreturnedregular30toprioritizefull102report; no cancellation.
- Thirdtransferpack complete18m45:70/102states,1401compactchecks,29original
  numericfailures; candidate20passesall, other11newstates failbroadcountgate.
  Refined8 complete34m58/all512pass/10checks, original512flags retained.
  Existingregularsparse11/14/19/20/21/22/24/26 now30min(max297flags), noQOS
  orintegrandchanges; actualfull512timings34m21–34m58, proportionalestimate
  notguaranteed. No runningjobshortened. State11nowrunning.
- All32originalentry statescomplete;448groupedcompactchecks/32768branchrows
  authenticate11052originalfailedcontexts over16384state-contextpairs, same512
  PSFidentities. Actualmissing29guard anddraw00missinglocaltablerecovery retained.
  48-workerpilot10 COMPLETE25m36/all512pass, original500flags/12checks retained;
  batch159.083GiB. Onlyunstartedstate9 replaced61242565(debug30) after12source
  hashes/zero-runtimecancel/dependencyrepair; no broaderrollout. Existingref23
  (3flags) promoteddebug30; allotheroriginalfailure/remainingjobs preserved.
- Refined6 COMPLETE34m21/all512pass/10compactchecks,original512flags retained;
  initial27complete/512flags. Existing32-workerrefinementarraythrottle now6
  basedonactualfull512state34m41/34m21timings;1hlimits unchanged. Parentupdate
  returnedfinishedchild6error butread-only13/14/31 confirmeffective6; original
  failure/poststate receiptsretained, norepeatupdate or48-workerrollout.
- Refined5 COMPLETE34m41/all512pass,original512flags/10compactchecks retained.
  Initial25/26 flags512/149; two-stateinitialpilot61239060 COMPLETE16m13,
  original30/31 flags512/420,20compactchecks pass,batch49.612GiB. Remaining
  refinements/fullposterior stillrequired; initial32nowcomplete; transferpack2running.
- Integrated docs/v3_1_result.md now synthesizes completed workpackages with
  explicitpendingtransfer/structure/domain/fiveplates. Sourcehashes and14local
  links recorded; notfinalacceptance. Replacependingsectionsonlywithactualresults.
- Latest full suite: 179 tests pass (6.06 s). Seventeen English plates have
  actual visual acceptance: paired 62, calibration 32, latent 1574, candidate
  138, photometry 165, external 104, radial 1823, response 636, candidate modes
  3613 and domain evidence 1309 artifact/source checks; response compact 351.
- Reproducible scripts/v3_1_transfer_compact_check.py validates94states with
  1881checks, preserves33failures; default102guardrejectspartialinventory.
  Standalone48-worker exactrefinementadapter prepared (288GiBplanningbound,
  sameoriginalcontextfunction); replacementpilot10/61240361running after
  zero-runtimecancellation/sourceproof/dependencyrepair; nowCOMPLETE25m36/all512pass.
  Resume2/61239061 COMPLETE5m40 andrefined12 COMPLETE2m15:512pass each,
  original512/12failures retained;10compactchecks eachpass. Entry21–24 complete
  with103/133/3/297originalflags. Remainingtransferpacks2–4 promoteddebug30
  frommeasured17m41/18m31, max2packednodes unchanged. Originaltransferarray
  throttle2→4 afterprovingonly5/17liveandzerorunning;actualmaxwriters2, no
  resubmission. Schedulerrootcauseunconfirmed; receiptsretained.
- Prepared scripts/v3_1_remaining_figure_check.py forfivependingplates: exact
  source/panelcoverage andPNG300dpi/EnglishPDF. Onlyparse/import/missing-input
  guards tested; actualfigureintegrityandvisualreview remainrequired.
- Candidate mode decomposition/report complete; preserve all grid levels,
  basin responsibilities and inherited evidence failure. No mode rerun needed.
- External report, source evidence maps and relative screens complete. Finish
  scientific domain adjudication using remaining actual transfer/structure
  outcomes. Do not reopen/reselect or regenerate predictions. Retain all 72
  common-cohort score failures and all inherited ensemble flags.
- Pending short response stages replaced by4sequential groups61229862 after
  all60oldpending stages provedcancelled/unstarted;68complete stages retained.
  Originalall64report61221949 nowcomplete; all4resourcegroups complete.
- All 32 independent SED refits are complete (last exact resume 61227619_7).
  Registered old/new probes/refinements/report complete. Preserve all64inventories,
  original/pooledfailed normalization andnewESS1.023628; no addedresponse search.
- All102conditionalpopulationstates andallthreeensembles complete; allmain
  evidence gatesfail, threecompactchecks each229pass. Finishfrozen sky-transfer
  61238249 plusretained61231554_5/_17/_24/_25 andreport61231678.
  Firsttransfergroup61231554_0 complete17m47/fourbroad states,6000/935PSF;
  88compactchecks. CandidatecutjointESS4.57–25.95 fails support despite conditional
  numericpass; these arein-sample, notheldoutfitresults. Twoactual-source
  transferplates prepared; notrendered/accepted.
  Historicalcompleteddependencies repaired with52actualstate hashes; preserve
  initialrejectedreportdependencyjournal. Native/joint/entry matrices,
  all candidate states and fixed-high-q joint counts are already complete.
- Radial implementation, all176/352outcomes, correlatedsummaries andthree
  visuallycheckedplates complete. Preserve failedscience andalloriginals;
  sourceblocksmostlysingleton andentrylocal-dust approximation remain explicit.
  No additional radialrun is required by the current protocol.
- Finish all 32 structure native/entry/refinement states, including exact entry
  refinement draw0 nowcomplete61228245/all512pass. Replacementarray61229422
  handlesstates2–31 withvalidated32workeradapter, originaldraw1completedall512pass.
  Pendingrefinementlimits1h/max3nodes basedon1548.72s exactpilot; native/refined
  all32complete,490originalfailures/0refinedunresolved. Entrystate3 has24initialfailures. Existingdependentchain61232760/61/62 nowqueued
  foractualshared-posterior/count inference andfairprior-withdrawn baseline;
  initialsubmissionreceipts alone are notscientificcompletion.
- Apply existing probability <.003, count-mean <.5, sampler/bridge/importance
  support and evidence gates; preserve failures rather than changing thresholds.
  Main finite-ensemble evidence remains failed despite stable per-object scores.
- Complete-evidence domainjoin61233366 submittedafterok61231678/61232762;
  one6-panel status/ambiguity figure prepared. Writtenreportgenerator
  scripts/v3_1_domain_result.py nowrequiresall12actualsource markers, per-object
  mode/identity closure andactualtransfer/structurewrittenreports; missingjoin
  failsbeforeanyoutput. Onlyparse/import/missing-input guard tested; actual
  scientificinterpretation/figureacceptance stillpending. Actualexecution requiresfull102
  transfer andfull32structure reports; writtenreview/visualinspection stillpending. No new
  gates/model/labels; failedmain cannotbe promotedby stablemode sidecars.
- Complete scientific use-domain adjudication, source tables, English figures,
  visual QA, result/ledger/README/AGENTS and safe milestone sync. No deployment,
  million-object rerun, timers or subagents; large scientific products stay NERSC.

## Completed v3.0 research delivery

Full authorized review: docs/pro_review_b59e18b.md (14,327 characters).
Results: docs/v3_0_result.md;42-plate atlas:docs/v3_0_atlas.md;
actual final acceptance:manifests/v3_0/delivery.json.

- [x] A — native PSF/exact identities/roles; full reference sources, independent
  eligibility and retained non-1 SpecDis flags/missing labels.
- [x] B — continuous empirical native-flux SED likelihood and multiple branches,
  weak/negative/missing/censored bands, constrained binary/dust/discrepancy channels.
- [x] C — source calibration,32full reference refits, selection-aware shared target
  posterior, reweightable joint products and correlated int64 counts.
- [x] D — same-identity QRF/band/probe comparisons, continuous recovery and migration,
  distance/class/PDF/Gaia/count diagnostics with scientific failures preserved.
- [x] E — once-only post-freeze reservation/new-sky/independent-scale validation,
  matched-response prior comparisons and NGC288 marginalization example.
- [x] F — all1183990qualified candidates exported; shared fullcatalog counts,
  coverage/masks/prior sensitivity,42inspectedEnglish figures and final audit.

Final job61163478 COMPLETED47s andpasses1309actual-array/artifact checks.
73focused tests,363A–D production checks and336figure checks are separate evidence.
All7original conditional numerical flags remain:6reference/1MC. Same-model
refinement sidecars for sixreference flags converge; originals are not replaced.
One prospective MCflag andone original nominal-reference flag also remain explicit.
Shared counts are conditional transfers from26000targets; effective responseESS1
and real/scenario calibration failures preclude universal accuracy claims.
Near expectation1109714.6; selected qnear>.95 count1013855; expected leakage
0.27361% is model-based, not independent population contamination.

No outstanding authorized production or scientific deliverables. Preserve the
frozen outputs and deployed v1. Next scientific changes require a new user scope;
do not create timers, agents, new models, reruns or deployment replacements.
Final git synchronization is checked before closing the chat goal.

## Completed v2.6 observed-catalog inference
Full requirements: docs/pro_review_39e6e4f.md; results docs/v2_6_result.md.
- [x] A: same192catalog per-star joint probabilities and correlated latent counts.
- [x] B: known/same/transfer controls, interval endpoints and numerical reachability.
- [x] C: nested256/768/2048sources; finite-noise checks and evidence-weighted response uncertainty.
- [x] D: cal-only source-conditioned signed-Gaia luminosity offset/scatter; same-identity latent grz transfer. Initial/fine grids preserved.
- [x]1648actual checks,16focused tests; nineEnglish figure sets and requirement ledger.
Final validation61130114COMPLETED0:0. No active science jobs or timers.
Results improve observed counts but response-transfer/coverage and source-dependent
calibration limitations remain; no adoption or calibrated-real-population claim.
3250prospective outcomes stay closed. Next scope requires new user/Pro direction;
no automatic W1, model family search, full-catalog expansion or deployment.

## Completed v2.5 target-population inference
Review docs/pro_review_de4904b.md; frozen scope docs/v2_5_protocol.md.
- [x] Outcome-blind red controls, signed-Gaia original metal pairs, class band gains.
- [x] Independent-template selection-normalized four-parameter response prototype.
- [x] Same-catalog point/PDF/population recovery and uncertainty/support audits; numeric-only2^18/2^17refinement completed.
- [x] Keep3250prospective outcomes closed pending model/selection acceptance.
- [x]4tests,1966allocatedcomputechecks,8inspectedEnglishfigures,80deliverychecks and report complete; final safe sync recorded inWORKLOG.

Final61127630 COMPLETED0:0; allstagejobs ended. Distance-fraction mean recovery
improves, but2–5kpcinterval undercoverage and physicalcomposition errors reject
use as calibrated population counts. Models/deployment unchanged;3250outcomes
stillclosed. Numerical-only refinement preserved catalogs and QRF bytes. Further
scientific development follows user/Pro review; no automatic next model or timer.

## Completed v2.4 frozen-model scientific-use audit

Review: docs/pro_review_0f9c412.md. Results: docs/v2_4_result.md;
English figures: docs/v2_4_atlas.md; requirement/limit ledger: docs/v2_4_requirements_result.md.
- [x] Same711/147anomaly audit;196SPDist intersections including60lowPIT, source conflict retained; source-separated2055pairs and repeated-control/sky uncertainty.
- [x]3545observable/2589near application objects, exact frozen joint counts and reference coverage; fitted identities separated in Gaia diagnostics.
- [x] Same33251development IDs, cal-only global/r/coarse-sky priors, matched population stress, band gains, reliability and clustered uncertainty.
- [x]224held-out empiricalSED templates/4480rows;4fixed populations, point/PDF/joint counts, teacher and noise-support limits. Population-count recovery remains unreliable under tested shifts.
- [x]3250new identities across13existing fields reserved with nativePSF and no historical overlap; reference outcomes/predictions unopened.
- [x]1898compute checks,4focused tests,167delivery checks;11English PNG300dpi/PDF/CSV/JSON figure sets visually reviewed.

All task jobs ended. Final validation61125423 COMPLETED0:0; first full validation61123659 and reservation61124175 also completed. Earlier writer failure and deliberate I/O migration preserved in execution/jobs.json and WORKLOG.
No new model, deployment, full-catalog prediction, automation or agents.
Scientific limits remain: source disagreement, independent metal-poor scale,
true application composition, bright-template noise/selection extrapolation,
new-sky/system independence and faint-population evidence. Prospective IDs are
NOT a completed final validation. Next scientific scope follows user/Pro review.

## v2.3.1 correction completed

Review docs/pro_review_569d6ff.md; frozen amendment docs/v2_3_1_protocol.md.
- [x] Correct shared-g sign; independent known-SED and non-diagonal band-space mixture tests,10focused tests total.
- [x] Preserve fitted179EM XD/QRF files and all QRF outputs; archive1336old candidate/derived files; candidate version/hash invalidation; recompute original guards.
- [x]61081256 COMPLETED0:0;8124production checks. Corrected candidate still fails full-Gaia/near-contamination guards; no adoption.
- [x] Replay42076reserved IDs and distance qualification exactly; no unexplained clean/unknown use.3502upper20/94untruncated eligible Gaia. Supplemental raw/conditional scores at32quadrature; forward simulation uses equal-object tails.
- [x]61081737 COMPLETED0:0;3316correction checks. Far full-PDF mass, SPDist evidence, original2055pairs and split-pair counts, dev-only/cal+dev coverage.
- [x]16English plots/report completed and visually reviewed; final source/format validation and three-endpoint synchronization recorded in WORKLOG.
No new blind claim, refit/search/deployment, timer or full-catalog run. Near
PIT low tail persists after specified selection correction; independent source
scales and representative target population remain unresolved. Preserve original
cal rules separately from supplemental selection analysis.

## Previous v2.3 delivery — candidate inference superseded by sign correction

Full review: docs/pro_review_83e1fe8.md; immutable protocol: docs/v2_3_protocol.md.
Results: docs/v2_3_result.md; nine-plate atlas: docs/v2_3_atlas.md;
requirements and remaining scientific limits: docs/v2_3_requirements_result.md.

- [x] Cache-only noise baseline correction and outcome-blind inventory:173chosen,2952weak/644SNR>=5Gaia,42076reserved IDs.
- [x] Fixed near scope and original-owner266774PSF distance support,43914unlabeled;7math/scoring tests.
- [x] Fit/freeze one XD32 candidate and same-identity QRF before reserved outcomes;179iterations. Numerical tail/grid migrations preserve fitted model files.
- [x] Selected-distance, source-convolved likelihood, full signed Gaia, matched poor-metal, far branch/reference and weighted target-coverage diagnostics.
- [x] Final61079683/61079765 COMPLETED0:0;8123actual artifact checks,9English PNG/PDF/CSV/JSON plates and visual QA.
- [x] Fixed cal rejects candidate: full-Gaia score and near-contamination guards fail. Existing deployment unchanged.

No active scientific jobs, full catalog, monitoring or subagents. Scientific
validation remains incomplete: r<20 reserve has only1broad-error chosen
reference; independent poor-metal scale, far-giant membership and faint/noWISE
population evidence remain unresolved. Next work requires new user/Pro scope;
do not search additional models or reinterpret reference contamination as the
application population rate. Final delivery validation and sync are recorded
in WORKLOG and manifests/v2_3.


## Completed objective: v2.2 selected-distance and actual-application audit

Latest review: docs/pro_review_fbb8ea0.md; frozen protocol: docs/v2_2_protocol.md.
Results docs/v2_2_result.md; seven-plate atlas docs/v2_2_atlas.md;
full evidence/limits ledger docs/v2_2_requirements_result.md.

- [x] Endpoint repair: actual max1+4.44e-16; all1280 reliability denominators conserved; chosen308cal/379dev restored, thresholds unchanged.
- [x] Same expanded_A and actual baseline_v1 ABC routes on26000PSF application objects;204full-PDF chunks, native grids retained; max catalog-mu error9.54e-7mag.
- [x] Selected distance/FeH/quality/sky, full weak/negative Gaia, population shift, unused-identity versus new-sky audit.
- [x]256unfitted development templates,759matched targets,4shared-noise repeats plus clean control; full Sobol128 PDFs, paired source-cluster uncertainty.
- [x] Far10–20/20–50 per-reference shapes, conditional class frequencies, training/app selection, missing-label luminosity support and retained Gaia extremes.
- [x]3559actual compute checks,6targeted tests;7EnglishPNG/PDF plates inspected; 107compact/format/link checks and final three-endpoint sync.

Successful jobs61070873/61071273/61072326/61071274/61071628.
Earlier writer failures61071166(native grid),61071786(sky field) explicitly
migrated with hashes; predictions reused. No active jobs, timers or agents.

Scientific gaps remain: high-q application53.25%weight has<20references per
observable cell; high-q nearby metal-poor dwarfs remain biased.Corrected in v2.3:42076unused
identities have173chosen distance-evaluation labels (3596signed-parallax
eligible, not all weak); only5buffered-sky identities, membership/scale checks pending.
No independent final-validation claim, new candidate, calibration, deployment
or full-catalog rerun. Next work follows user/Pro review and new target evidence.

## Completed objective: v2.1 near-selection and joint-reference probability audit

Corrected Pro review from「评估发表进展」on2026-11-10 archived in
pro_review_602727c_near_probability.md. Result: docs/v2_1_result.md;
six-group/eight-plate atlas: docs/v2_1_atlas.md; full acceptance ledger:
docs/v2_1_requirements_result.md. Frozen protocol remains unchanged.

- [x] A: strict PSF independent qualifications, full near/far/boundary statistics,
  weighted actual baseline_v1 application sample (13 DECam fields).
- [x] B: exact frozen leaf class/source contributions,97650 full PDFs/763chunks,
  shared-g Sobol128, original marginal/quantile and hash reproduction.
- [x] C: matched+broad classification probes, covariance neighborhoods, target-noise,
  weak signed flux, near reliability/full probability, FeH/sky/nonstellar controls.
- [x] D: one prespecified same-PSF-identity W1 class-partition candidate, cal rejects;
  no second candidate or deployment. Report full development harms as well as gains.
- [x]6511 compute artifact checks,12926 compact/figure/link checks,7 focused tests;
  all eight PNGs visually reviewed and representative PDF rendered/inspected.

Final candidate61069324 COMPLETED0:0; final extreme-tail quadrature/artifact
validator61069709 COMPLETED0:0. No active jobs or timers. Final milestone source
sync verifies local/origin/NERSC equality. Keep all large caches on NERSC.

Outstanding SCIENTIFIC evidence: independent final validation is NOT achieved.
Reservation leaves5 unused IDs/2tiles, pending system/cluster membership and
independent-reference audit. Faint/no-WISE evolved, nearby metal-poor dwarfs,
far giants and representative PSF nonstellar/application contamination require
new evidence. Current development test never becomes a final test by relabeling.
Next: user/Pro review of these results; no automatic further modeling/application.

## Completed objective: replace five distance comparison plates

Job60947091 COMPLETED0:0 in29s,nid006606. Five existing PNG/PDF/CSV/JSON
sets replaced at their original main/appendix paths. Main + attached4:1
residual (-100 to100 percent) + central histogram; obsolete full panels removed.
Each population/reference comparison has its own one-pass x/y distance3sigma
mask,shared by its three panels. Final user clarification is authoritative.
Real all counts: Gaia3207→3123,SpecDis47335→45892,SPDist37317→35820.
Three focused tests,source hashes,per-comparison mask/count checks,formats and
all five visual reviews pass. Layout-only compact-bin re-render fixed sparse
colorbar tick overlap without changing any CSV byte. Atlas still1050 figures,
now3673 panel/selection conservation checks. Current delivery hashes updated.
Original scientific scores and49-object anomaly audit remain unchanged.
Next: user/group review; no new scientific computation or monitoring scheduled.

## Completed objective: v2.0 revised atlas and Gaia anomaly audit

Latest Pro review of6ee0df1: docs/pro_review_6ee0df1.md.
Result: docs/v2_0_revision_result.md. All1050 figure sets embedded in27
volumes via docs/v2_0_revision_atlas.md. English-only transparent figures.

- [x] Terminology/methods,975 band combinations,55 refreshed existing figures,
  15 temperature/Kiel views,one parallel reference/application flow,4 anomaly figures.
- [x] Job60941065 COMPLETED0:0 in24:06 on nid004940; source39ddedb.
  Previous60940195 unsigned missing-fill failure diagnosed and regression tested.
- [x] Counts47756/3207/1163/49,3687 panel conservation checks,frozen inputs retained.
- [x] All1050 PNG/PDF format checks;270 contact-sheet visual reviews plus detailed
  primary views. Local compact-only flow polish shows parallel application outputs.
- [x] Complete linked CSV/JSON/PNG/PDF delivery and updated result/protocol/memory.
  Final milestone synchronization uses scripts/sync.py push and endpoint SHA checks.

49 anomalies support a low-luminosity/WD candidate explanation,not confirmed
classifications. Gaia/native position agreement is not independent matching
validation. Equal-width residual bins show no marked zero-centered hole.
Original frozen references/models/predictions and16 historical nonPSF exceptions
remain intact. No new training,full-catalog prediction,monitoring or subagents.
Next: group-review discussion; further scientific work needs new scope.

## Completed objective: Photdis v2.0 stage freeze

Scope/methods/status/atlas and docs/v2_0_requirements.md cover the long Pro
review. Four base statistics table families,19 normalized views,55 figures
with per-figure CSV and metadata. Full TYPE and PSF audit complete; percentage
metrics recomputed per star.391 actual checks; focused metrics test and
synthetic/compact figure smoke pass. All55 PNGs reviewed, PDFs parsed and
representative PDF pages rendered. Next step is group-review discussion.

## Completed objective: fixed weak WISE flux contrast

Result: docs/baseline_v1_6_wise_result.md. Exact native signed flux, 122217
shared templates and 97705 matched cal/test rows. Three real-weak primary
contrasts show net losses; 28 evolved stars with valid W1 and weak W2 remain
severe failures. Weighted either-band measurement coverage is 50.41%. Close
this prototype with its negative evidence and three inspected figures.

## Completed objective: expanded A apparent-magnitude contrast

Result: docs/baseline_v1_5_r0_result.md. One fixed QRF contrast, 317003 train and
97705 aligned cal/test objects. Refined signed Gaia and paired spatial analyses
complete. Evolved severe tails 96.56% to 89.85%; dwarf 1.95% to 2.16%.
Metal-poor dwarf bias persists. Coverage uses actual class and distance
eligibility and separates reference types. Five reviewed figures use Pal 5 New
Step 5 styling. Current development evidence does not support adoption.
Further work requires a separate authorized experiment and independent test.

## Completed objective: v1.4 scientific diagnostics (2026-11-05)

Authorized latest Pro review: docs/pro_review_46cb68a.md. Protocol:
docs/baseline_v1_4_diagnostics_protocol.md. Freeze all models and primary scores.

- [x] Identity-safe per-star joins and refined Gaia reconciliation.
- [x] Distance, CMD, color and spatial scientific figures, then spectral/PDF diagnostics.
- [x] Paired gain/loss with sky-cell bootstrap; parent support and joint WISE cuts.
- [x]1943 actual checks,4 focused tests,29 figures and scientific interpretation.
- [x] Report and delivery manifest prepared for safe three-endpoint synchronization.

Result: docs/baseline_v1_4_science_result.md; complete atlas:
docs/baseline_v1_4_science_atlas.md. Delivery metadata records final rendering
and inspected figure hashes. Source sync verifies local/origin/NERSC equality.
Evolved distance failures persist inside support and across populated cells;
metal-poor dwarf distances are biased high. Overall paired gains are small and
limited to9 test cells; WISE quality/depth losses strongly overlap.
Next scientific scope should target branch allocation and metal-poor dwarf
residuals with an independent final test. Do not automatically train, change
deployment, rerun completed stages or create timers; await new authorization.

## Completed objective: v1.4 broader references and native i experiment

Scope: docs/pro_review_5cf8f6d.md and docs/baseline_v1_4_protocol.md.
Result: docs/baseline_v1_4_result.md; delivery: manifests/baseline_v1_4_delivery.json.

- [x] Full MWS/SpecDis/SPDist and1600 southern sweeps; original extraction hashes retained.
- [x] Exact IDs, zero IDs, duplicates/conflicts, VAC-only and targeted epoch-aware matching.
- [x] Independent class/distance/weak-parallax eligibility; new geometry and historical identity isolation.
- [x] Original26000 parent i/WISE coverage with original weights and raw flux provenance.
- [x] Frozen v1/v1.3 ABC scores on genuinely new sky before any new fit.
- [x] One fixed-algorithm historical-plus-new-data ABC comparison, same evaluation identities.
- [x] Same-ID grz/griz class and distance experiments with shared4D Sobol128 errors.
- [x] Convergence-controlled full signed Gaia scores in separate immutable sidecars.
- [x]3331 actual artifact checks,26 focused tests and5 visually inspected scientific figures.
- [x] Scientific interpretation, persistent memory and final milestone prepared for safe three-endpoint sync.

Controls60799511 COMPLETED0:0 in34:36; final execution/validation manifest records
all job outcomes. Local/GitHub/NERSC commit equality is checked by scripts/sync.py
at delivery. Large catalogs, models and predictions remain on NERSC.

New train138080 class labels and153403 distance objects; new test42964 class
labels. New data gives modest/mixed gains. Matched i improves AP0.2729 to0.2823
and overall tails9.71% to9.57%, but evolved distance tails remain96.52% on1064
same objects.18–20/no-W1 AP worsens; broad22–25 validation remains absent.
Retain deployed v1. No model family search, candidate retuning, full-parent rerun,
subagents or automatic deployment. Both automations remain deleted.

Next scientific step: review these fixed results and authorize a new scope if
needed; do not automatically extend experiments or resubmit completed jobs.

## Completed objective: v1.3 independent label entry and fixed evolved candidate

Source:docs/pro_review_ebf37f8.md; protocol:docs/baseline_v1_3_protocol.md.
- [x] Version independent eligibility; A adds6287 training labels,395 evolved.
- [x]135-cell owned-parent/reference coverage; no22–25 cal/test class evidence.
- [x] Six fixed forests on expanded labels; matched train/evaluation identities.
- [x] Frozen predicted-evolved B/forest candidate, contaminants and Gaia reuse.
- [x] Job60775835 COMPLETED0:0;93 actual checks,18 tests,3 inspected figures.
- [x] Report and memory ready for final scripts/sync.py push endpoint verification.

Result:docs/baseline_v1_3_result.md. Model change is small/mixed. Candidate
chosen-risk59.18% vs same-star wide49.66%; no replacement. Large files remain
on NERSC data/baseline_v1_3; no distance refit/full-catalog prediction rerun.
Next: acquire labels in identified faint/no-WISE/large-error cells, retaining
new unseen sky for validation. Existing-field label release does not fill that
observational gap. End this fixed candidate attempt; no threshold retuning.


## Completed objective: baseline v1.2 color separability and distance decisions

Source:docs/pro_review_dbeac65.md; protocol:docs/baseline_v1_2_protocol.md.
- [x] Recover frozen protocol, classifiers, cal workpoints and same-training controls.
- [x] Fixed-parent coverage and full observed-parallax checks completed.
- [x] Job60731324 COMPLETED0:0;2364 actual checks pass;16 focused tests rerun.
- [x] Inspect ten figures, publish compact evidence and final result.
- [x] Final checkpoint prepared; scripts/sync.py push verifies identical local/GitHub/NERSC
  hashes before the continuation Goal closes.

Resumed2026-11-03 in chat01a1a092-2ca6-7512-922b-ac73315e6657.
Old chat01a18087-3d26-7e63-824a-fb325571b692 archived/interrupted; old Goal paused.
Recovered clean three-endpoint1de42a4. No new training/scoring/full-catalog run.
Preserve v1 deployment. C evolved recall35–46%, known-reference precision39–42%
at cal2% dwarf budget. Complete Gaia evidence disfavors hard main-branch distances.
Cal-passing single groups are not uniformly reliable: A/evidence evolved failure
is100%; B/forest has9.52% evolved failures on63 selected test references.
Outstanding scientific limits: reference composition and real faint/no-WISE
validation; no Gaia r>=22 objects. Do not launch additional experiments.


## Completed objective: baseline v1.1 fallback and learned weights

Source:docs/pro_review_97c2775.md; protocol:docs/baseline_v1_1_protocol.md.
Result:docs/baseline_v1_1_result.md; detailed risk and actual validation in manifests.
- [x] Recover97c2775, archive complete review and establish detailed Goal.
- [x] Fixed same-fallback oracle decomposition on existing references.
- [x] Training-only spatial cross-fit evidence and one jointly supervised gate.
- [x] Cal-only adoption and same-source/same-training development scorecards.
- [x] Prediction-state risk/retention and observed-parallax validation.
- [x] 470 actual checks,11 focused tests,8 inspected figures and final report.
- [x] Final milestone prepared for scripts/sync.py push; verify identical three
  endpoint hashes before completing Goal.

All six gates fail; retain deployed wide v1. C evolved oracle tail13.91% becomes
15.04% with full fallback; new gate98.87% versus wide80.08%. A fixed high-weight,
narrow-width prediction group does not improve on the old model's same sources.
Preserve v0/v1 catalogs and Sobol128. No full-catalog rerun occurred.

Training60727000, sampling60727153, acceptance/render60727969 and final
report-only render60728246 completed. Fixed samples:26,000 parent/7,672 Gaia,
including1,800 negative parallaxes. No Gaia validation at r>=22. Reference-risk
and weighted sampling coverage do not establish the unlabeled parent purity.
Reproduction audit60727875 bounds CDF rounding below4.02e-7; wide is exact.
Next global priorities below remain unstarted; do not launch another local audit.


## Completed objective: baseline v1 branch, numerical and depth comparison

Full review: docs/pro_review_494b12c.md; fixed protocol: docs/baseline_v1_protocol.md.
- [x] Recover clean494b12c, read full review, create detailed Goal and protocol.
- [x] Stratified convergence sample and analytical error-propagation correction.
  3000 sources; Sobol128 improves tails but fails strict layer tolerances; flags retained.
- [x] Whole-reference branch-mass audit, conditional/oracle/photometric models.
- [x] Cal-only selection, same-training comparisons, fixed flux-depth experiment.
- [x] One full-parent application,321 actual checks,8 reviewed figures and report.
- [x] Final milestone prepared for three-endpoint sync; verify hashes before Goal completion.

Preserve v0 and use data/baseline_v1; no cluster or algorithm-family search.
Six branch families, reference evaluations and fixed flux-depth experiment complete.
Cal retains wide QRF and B/C upgrades; no all-class quality cut passes. One full
application60721565 produced9,390,518 rows; validation/report60721672 passed321
checks. Result:docs/baseline_v1_result.md. All eight final figures inspected.
Preserve numerical-convergence and research-quality failures as actual findings.
Final delivery uses scripts/sync.py push with identical endpoint hashes.

## Completed objective: unified multi-population baseline v0

Full review: docs/pro_review_60f65d5.md; execution: docs/baseline_v0_protocol.md.
- [x] Recover endpoints, read complete review, detailed Goal and protocol.
- [x] Unified optical parent, broad labels and identity-safe spatial split.
  9,390,518 unique parents; training A/B/C163600/149743/136078.
- [x] Fixed QRF A/B/C plus same-training controls; cal-only routing/intervals.
- [x] Whole-test validation and full existing-field internal predictions.
- [x] Nine figures, final scorecard,3 mathematical tests and206 artifact checks.
- [x] Final milestone prepared for scripts/sync.py push; endpoint hashes verified
  by that delivery command before completing the Goal.

Result: docs/baseline_v0_result.md; schema/runbook: docs/baseline_v0_catalog.md.
8,899,334 DECam internal predictions;491,184 north rows marked no-native-model.
Matched Gaia NMAD A/B/C=.266/.242/.218mag; W1/W1W2 coverage31.7%/17.8%.
Recommended Gaia relative NMAD10.4% is dwarf-dominated. Low-gravity SPECDIS
subset298 has81.9% |delta-mu|>1mag; r>=22 validation is inadequate.

## Next global priorities (not started)

- Investigate population degeneracy and reference composition using the fixed
  split; report evolved/transition failures separately from overall scatter.
- Establish validation for low-precision/no-WISE objects that dominate the
  optical parent, before choosing scientific-product quality thresholds.
- Preserve v0 artifacts and version any follow-up. Do not automatically resume
  cluster searches, model-family searches or treat northern rows as predictions.

This supersedes the64-warm-member phase task as the project main line.
Internal uncalibrated field catalogs are authorized. Historical models remain frozen.


## Completed objective: Pal14 parameters and fixed120-spectrum observational coverage

Full review: docs/pro_review_68eb2f3.md; protocol: docs/giant_parameter_audit_protocol_v1.md.
Result: docs/giant_parameter_audit_result_v1.md.
- [x] Full review, bounded Goal and unchanged22 predictions.
- [x] Literature/DESI/Gaia/DECam, images and conditional MIST evidence.
- [x] Exact120 spectra/120 stars, directed DR11 and complete coverage census.
- [x] Scope/denominator corrections and evidence-based route decision.
- [x] Five actual checks, syntax checks and four inspected figures; final milestone.

Pal14 identity association is strong; temperature/gravity provenance conflict
remains unresolved. DESI member spectra exist in18/159 linked systems only.
Outside K window112 stars are5000–7000K;64 core grz DECam stars in10 systems
have zero overlap with either registered target color range. Four northern
warm color/luminosity overlaps are all metal-poor NGC2419, not DECam K anchors.
Current K absolute calibration remains evidence-insufficient; close this census.
No new scores,141-system expansion, changed window or model tuning.

## Deferred local evidence direction (superseded as main task)

Register the64 DECam warm members as resources for the existing5000–7000K
warm-evolved branch. Before any new distance fit, define independent phase
evidence, actual target support and system-level development/test roles; retain
historically unblinded systems as such. Do not silently broaden the K branch.
Preserve cold-v4, blue/metal-poor main-sequence and all prior RGB results.
Final milestone delivery uses scripts/sync.py push and endpoint hash verification.

## Completed objective: fixed BV21 multi-system external reference batch

Source: docs/pro_review_35009d2.md. Detailed protocol:
docs/kgiant_reference_batch_protocol_v1.md. Goal registered; no model retuning.

- [x] Read full review and freeze task boundaries.
- [x] Versioned162-system distances/methods and170-system membership linkage.
- [x] Prior-result latitude/support/quality/reference-scale addendum.
- [x] Fixed three-star image or independent-photometry check.
- [x] Member-first full DESI type audit, directed DR11 photometry, freeze coverage.
- [x] Fixed A/prior and same-star A/B scoring, per-system uncertainty and limits.
- [x] Validate, inspect figures, final report, checkpoint and three-endpoint sync.

Result: docs/kgiant_reference_batch_result_v1.md. Fixed parent complete;
one new Pal14 optical-only member plus three historical Terzan7 members,
22 configurations /4 stars /2 systems. Pal14 grz centers55.68/48.22kpc
versus73.579kpc reference. No new W1 pair; no field calibration claim.
Next evidence need: joint color/luminosity/phase/reference coverage and usable
bright photometry, not automatic new clusters or model tuning. Final milestone
is delivered through scripts/sync.py push with identical endpoint hashes.

## Completed objective: fixed six-object external K-giant check

Full review: docs/pro_review_a399c3e.md; preregistration:
docs/kgiant_external_protocol_v1.md. Preserve historical results.

- [x] Read full review, create detailed Goal, register six-object scope.
- [x] Correct190-object coverage and empirical interval interpretation.
- [x] Resolve literature identities or record unverified links; independent flags.
- [x] Allocated DR11 five-position match; three accepted stars,18 fixed mixtures.
- [x] Shared-scale sensitivities, per-star distributions and explicit failures.
- [x] Six focused tests, ten actual checks, four inspected figures and result report.
- [x] Final report, provenance and milestone prepared for three-endpoint delivery
  through scripts/sync.py push; verify endpoint hashes at delivery completion.

No classifier/bandwidth tuning, no expansion beyond Pal12 one + Terzan7 five.

## Completed objective: actual K-giant teacher audit and conditional distances

Source: docs/pro_review_1689ec2.md; detailed frozen scope and acceptance:
docs/kgiant_distance_protocol_v1.md. Preserve all previous components.

- [x] Read complete review; create detailed Goal; update AGENTS checkpoint,
  cold distance ranges and warm catastrophic-tail table.
- [x] Register finite teacher/independent-reference/conditional-mixture experiment.
- [x] Exact selected190 evidence, common teachers, errors/flags/coverage/parameter trends.
- [x] Independent member/phase/luminosity support census; scoped external decision.
- [x] All-qualified K teacher training; whole-field A/B/no-color distributions,
  paired selected evaluation, shared-g errors, SPECDIS width sensitivity.
- [x] Validate formulas/identity isolation/real summaries; inspect five main figures.
- [x] Registered Pal12 feasibility completed60678349:372 high-probability members,
  300 clean grz; zero current low-dust K references. No model scoring or relaxed cuts.
- [x] Final report, decision and provenance: docs/kgiant_distance_result_v1.md.
- [x] Final milestone delivered through scripts/sync.py push; verify identical
  local/origin/NERSC commit hashes at the end of delivery.

Current decision: retain teacher-scale conditional mixtures as research prototypes.
Selected common teachers39, median SP−SPEC modulus−.274mag; actual189 SPECDIS
median fractional error29.2%. Selected grz/grzW1 NMAD.511/.479mag (SPECDIS);
W1 score gain small, scatter improvement not robust across fields. No independent
absolute calibration, no field-giant distance product; raw coverage is not truth.

Next bounded evidence requirement: DECam K giants4200–5000K near FeH−1..−.5,
Mg<.5, with independent distance/phase evidence and usable bright photometry.
A different extinction regime or membership/phase proxy must be separately
registered, not quietly added to force Pal12 through this experiment. Keep the
multi-population project scope and all historical models frozen.

## Completed objective: controlled negative coverage, distances and matched-source CMDs

Full requirements: docs/pro_review_20261102.md; frozen protocol:
docs/giant_wide_protocol_v1.md; final report: docs/giant_wide_result_v1.md.

- [x] Recover endpoints, read full Pro reviews, save source and register protocol.
- [x] Correct denominator/score-gain/current-step wording; implement controlled fits.
- [x] Run allocated NERSC three-model whole-field train/cal/test experiment.
- [x] Aggregate same-star rates, paired field uncertainty and K calibration gaps.
- [x] Plot frozen phot-d versus Gaia-d / SPDist / SPECDIS with matched identities.
- [x] Inspect six figures and pass 13 focused tests plus 11 real-data checks.
- [x] Deliver report, result provenance and milestone to local/GitHub (`6bcf02f`).
- [x] NERSC authentication renewed; scientific distance results already complete.
- [x] Add three CMD figure groups using the exact cold/warm cached identities and masks.
- [x] Validate counts, inspect reference/model CMDs and deliver through three-endpoint sync.
  CMD job60670802 completed0:0 in17s; no new cuts or model fitting.

B-wide at same-model 1% calibration retains190/289 K giants versus B-K28/289;
actual dwarf FPR1010/98536. Bright r16–18 recall168/229 versus6/40 at r18–20.
Freeze the scoped DECam enrichment recipe; do not claim a high-purity catalog,
calibrated probabilities, or a single global threshold. End fixed model search.
Selected190:170 valid FeH (median−.85),39 SPDist/189 SPECDIS teacher matches;
34/39 and179/189 teacher Mg lie above the old lower-RGB luminosity range
(Mg<.5). Independent phase is unknown. No automatic lower-RGB extrapolation.

The above classifier-era next task has now been completed as the K-giant
teacher experiment at the top of this plan; consult its measured reference gap.
Do not repeat the completed FeH mechanism or widen fields without an evidence gap.

## Completed objective: observable DECam giant candidate application

Followed docs/pro_review_8e26f78.md and frozen giant_candidate_protocol_v1.md.
Report: docs/giant_candidate_result_v1.md. Existing 14 fields only, original
models unchanged; broad DECam reference N=205699, W1 N=180306, common5 N=159103.

- Completed original-model broad scoring, W1-only matched A/B, temperature/class/
  support/brightness/W2 splits, descriptive ROC and nested train-only thresholds.
- W1 B inside-box score>=.5 retains 244/289 K giants but admits 20104/98536
  broad dwarfs. Train-side 1%/.1% FPR targets retain 31/289 and 6/289 K giants;
  actual outer FPR 2.32%/.132%. Do not promote this classifier as a high-purity
  field catalog. Only seven additional K giants lack W2: coverage remains weakly
  validated, no r>=20 K-giant performance claim.
- Observed T/logg-conditioned FeH comparison: common5 grz gain +.490 per star
  on six informative fields, CI [.294,.750], all positive; limited predictive
  evidence, not physical abundance calibration or independent distance prior.
- DR11 north11011 exists; sky-north naming is not camera identity. Northern
  expansion deferred until adequate same-camera fields are available.
- Allocated fit 60635993 and corrected summary 60636273 completed 0:0. First
  summary 60636037 failed on Wilson plotting roundoff only; fixed error lengths,
  preserved fit outputs. 18 focused tests, aggregate checks and figure inspection.
  Code/reports/aggregate manifests/plots delivered via git; row scores/models stay
  NERSC. No further threshold/model search in this completed round.

Next milestone: a separately registered observable temperature/class coverage
scheme that addresses warm dwarfs, intermediate gravity and unknown/nonstellar
risks. Then align phase/FeH/luminosity references to the selected population.
Current lower-RGB relation is not a field-giant distance product.

## Completed objective: existing-field giant selection and abundance information

Full source: docs/pro_review_a7b5c5d.md; frozen scope:
docs/giant_information_protocol_v1.md; conclusions:
docs/giant_information_decision_v1.md. All bounded tasks completed, including
uniform-control metadata, existing-field census, native-camera/identity isolation,
class and hidden-FeH experiments, whole-field comparisons, figures and interpretation.

- Allocated jobs 60634820/60634824/60634834 completed 0:0 on 2026-11-01.
  14 fields: 9,390,518 clean photometric records; 319 K giants / 21,555 dwarfs.
- Same-star DECam WISE>=5: A/C giant recall 74.8/87.9%, dwarf leakage
  25.0/9.3%. Northern camera has no other native training field. No giant
  validation labels at r>=20; scores are not full-catalog class probabilities.
- Optical FeH information passes the bounded marginal-baseline comparison;
  extra WISE information and the high-S/N panel lack stable cross-field evidence.
  Do not double-use colors as an independent FeH prior in distance inference.
- Validation: 24 focused tests, nine real-result consistency checks, four plots
  inspected. Machine-readable results include code/source hashes. Original
  compute summary preserved under ignored remote data before tracking small outputs.
- Deliver code, reports, aggregate manifests and figures through the repository;
  source catalogs, row scores and fitted objects stay on NERSC. No new fields,
  quality-cut tuning or RGB refit is required to complete this goal.

Next scientific milestone: independent phase, abundance and luminosity evidence
for the actual, mostly warmer K-giant candidates at r~16–18; quantify unknown
classes and selection before constructing a field-star distance model. This is
a new calibration task, not an automatic continuation of diagnostic field expansion.

## Completed objective: local conditional color information

Followed the full Pro 4c203ca review and fixed preregistered protocol.
Results: docs/cluster_rgb_local_color_result_v1.md and
manifests/cluster_rgb_local_color_v1.json. Interior N=278/163/69/25;
shared-nuisance marginal ratio, gr/grz/no-color, reference/three atom-weight
priors and two reddening widths completed. B varies by at most ~0.4%;
no-color posterior exactly recovers the local uniform prior. At reference FeH,
NGC5466/4147 grz W68=.370/.473 versus prior .680 mag; unknown FeH weights
move NGC4147 center by .171 mag. M3 retains endpoint sensitivity.

Numerical endpoint: 32 fits, 12 bounded refinements, 30/32 pass. M3 grz
reference E=.005 and unknown E=.01 retain log-target Rhat=1.01086/1.01638;
flags remain in tables, manifest and figures. Adaptive brightness integration
replaces inadequate GH quadrature while retaining originals. 34 focused tests
pass; all four final figures inspected; final job 60598989 completed 0:0.

Decision: retain the conditional RGB prototype and close this information
experiment. Next scientific milestone is to specify an RGB target population
and establish abundance/phase evidence before its luminosity and observation-
selection model. Actual calibration evidence is required for a broader support
range. No further automatic field expansion, curve search or chain extension.

## Completed Objective: external scales and shared-member RGB distance

Detailed goal: docs/cluster_rgb_scales_protocol_v1.md; final results:
docs/cluster_rgb_scales_result_v1.md and manifests/cluster_rgb_scales_v1.json.
Fixed 422/250/93/42 members, frozen curves and nominal bins retained. Completed
primary-source constraint table, coherent RR/reddening/FeH scenarios, two-color
geometry, finite common/differential reference budgets, shared-member inference,
no-color support control and wrong-independent comparison. External evidence
leaves scale ambiguity, so no reference refit or new regression is adopted.

Numerical endpoint: all 10/20-member subsets pass; full M3/NGC4147 pass.
Full M53/NGC5466 distance parameters pass, but log-target Rhat=1.0107/1.0122
misses the registered threshold. Retain those flags and end the bounded
refinement at 48000 steps/chain. Shared NGC4147 W68=.184 versus .038 for the
wrong-independent calculation and .197 for no color; selection is informative.
29 focused tests pass; four scientific figures inspected; all final Slurm jobs
completed. Programs/docs/small results/figures synchronize; catalogs/chains stay
on NERSC. No active compute or further automatic model search is required.

Next task narrowed by Pro 4c203ca: first quantify local conditional-color
information; defer the full selection/stage likelihood and reference hierarchy.
The current output is a conditional prototype; existing distance products and
scientific quality cuts remain frozen.

## Completed Objective: independent NGC 4147 lower-RGB validation

Followed docs/cluster_rgb_ngc4147_protocol_v1.md and the complete Pro
c1f5c59 review. Registered primary robust monotone grz and all frozen
controls before extraction; locked 42 common DECam sources before scores.
All five Mg bins populated, 14 stars at Mg>=2, zero training overlap.
Slurm extraction/scoring jobs 60535204/60535316 completed successfully.
Primary center -.205 mag, NMAD .187, cover68=29/42 and cover95=37/42.
Color score gain over uniform prior +.371/star; paired z NMAD gain .022
has 95% interval [-.047,+.134]. The matched 48-cluster shared generator
allows the measured center and .330-mag faint/bright contrast, with broad
predictive ranges. Unknown-FeH weights shift center by .240 mag. Full
results and decision: docs/cluster_rgb_ngc4147_result_v1.md.

## Completed Objective: lower RGB scope, shape and prior controls

Followed docs/cluster_rgb_lower_protocol_v2.md and the saved full Pro review.
Training is restricted before fitting to 422 M3 + 250 M53 stars, with same
IDs and 48 nuisance realizations for gr/grz and both curve forms. Restricting
the domain eliminates quadratic reversals. Development top-1% residual trace
fraction 30.3% triggered the fixed mixture control without dropping members.
On the already-unblinded 93 NGC 5466 stars, grz NMAD is .125/.130/.110 mag
for lower quadratic/monotone/robust versus .231 for v1 and .583 prior-only.
The robust faint-to-bright median contrast remains .241 mag and 68% cover
is 96.8%. Internal full-generator coverage is near its nominal value;
cluster uncertainty and prior mismatch remain explicit. This is explanatory
validation, not another independent test or a released giant distance product.
All detailed requirements, metrics and provenance are in the protocol/report.

## Completed Objective: registered DECam conditional RGB experiment

Status: completed as a bounded model and once-only whole-cluster transfer
test. Followed the Pro review of `7256423` and the frozen protocol in
`docs/cluster_rgb_conditional_protocol_v1.md`; parameters and whole-cluster
identity are recorded in `configs/cluster_rgb_conditional_v1.json`.

1. Register NGC 5466 as a new whole-cluster DECam test before inspecting its
   Legacy photometry. Use its independent RR Lyrae distance and a common
   RRab-to-UVES metallicity scale. Keep its source IDs out of all development.
2. Audit NGC 7006 as a new development system: gr and grz upper RGB locus,
   local color residual, magnitude/radius/image-quality dependence, shared
   distance/dust sensitivity and actual overlap with M53. M3 lower and M53
   upper/lower remain other development stages; M3 upper edge diagnostic only.
3. Fit low-parameter forward gr and grz conditional RGB relations only on
   observed shared magnitude ranges. Propagate shared-band covariance and
   whole-cluster scale uncertainties. Keep existing per-band optical quality.
4. Freeze new model and fixed priors, then extract/score NGC 5466 with an
   allocated NERSC job. Compare bands on identical grz-clean stars and report
   gr-only added coverage. Evaluate known-abundance and three fixed unknown
   abundance mixture priors, by magnitude bin and with the registered metrics.
5. Write the physical result and limitations, run focused validation, update
   WORKLOG, commit milestone and synchronize tracked code/docs/small figures
   across local, GitHub and NERSC. Large catalogs remain on NERSC.

Measured outcome: NGC 5466 supplies 93 supported lower-RGB stars on the exact
gr/grz common set. Adding z improves distance-modulus NMAD from 0.314 to
0.231 mag (paired-bootstrap improvement 95% interval +0.009…+0.140 mag).
The grz median residual swings by 0.686 mag between the registered bright
and faint lower-RGB bins; its 68% interval spans a median 1.50 mag and covers
95.7% of point references. There is no registered grz upper-RGB support.
The model remains a conditional diagnostic and is not a field giant-distance
product. Unknown-abundance priors shift its median residual. A coherent
0.01 change in cluster E(B−V) changes the median residual by 0.152 mag in
the sensitivity experiment. See the result report, manifests and figures.

## Completed Objective: native-system, band-matched RGB audit

Status: completed as a bounded observational audit. Followed the Pro review of `beb15b7` with a bounded
four-cluster experiment. Keep the frozen pooled A/B fit, NGC 7006 test,
cold-v4/XD and blue forward-parallax results unchanged.

1. Complete: confirmed each member's DR11 `RELEASE` and natural camera system against
   sweep files and official Legacy documentation. Separate southern DECam
   from northern BASS/MzLS in all new comparisons. Review the existing
   `MW_TRANSMISSION` dereddening and quantify common cluster distance/dust
   uncertainty. Apply a fixed RGB passband conversion only with a reliable
   source; otherwise leave native systems separate.
2. Complete: re-read the full matched membership parent on allocated NERSC compute.
   Build a new `gr_clean` tier with exactly the existing per-band optical
   requirements on g and r, including their mask bits and BAILOUT, without
   any z condition. Preserve the grz-qualified frozen rows for same-star
   comparisons. Report by cluster and Gaia-CMD stage: gr and grz counts,
   same-row z-only failures, changed crossmatch row, luminosity/color/radius
   and image-quality distributions; inspect M3's bright upper RGB explicitly.
3. Complete: recomputed and plotted empirical loci from exact fitted stage cores: M3 lower,
   M53 upper plus lower, NGC 2419 upper. Show the NGC 2419 lower/red-HB
   overlap as a separate diagnostic. Distinguish within-bin 16–84% width
   from local-curve residuals and propagated photometric errors, including
   shared-band covariance.
4. Complete: compared M3 and M53 at the same M_g within their common DECam range using
   fixed interpolation. Audit residuals against magnitude, cluster radius
   and measured image-quality quantities. Treat modulus and dust errors as
   whole-cluster shifts. Inspect NGC 2419 within its own native system.
5. Complete: documented measured results, limits of four-cluster coverage and the minimum
   additional system/stage/metallicity cells needed for a conditional
   p(color | M_g, metallicity, RGB, system). Register a new whole-cluster
   test before any subsequent fit; NGC 7006 has already been used. Keep
   WISE as an acceptance diagnostic. Targeted tests passed; code, report,
   small manifests and the selected figure are in the milestone sync.

Measured outcome: M3 upper RGB has 97 clean gr vs 2 clean grz candidates,
but remains confined near `M_g≈+0.4…+0.9`; M3–M53 same-DECam lower loci
have `+0.0275 mag` median `(g-r)₀` difference at common `M_g`. NGC 2419
upper has a `0.090 mag` local gr residual NMAD with a strong radius trend;
only 25 quality PS1 counterparts pass the northern-system cross-check.
See `docs/cluster_rgb_native_system_v2.md` and two new manifests.

Next: the new experiment above registers NGC 5466 for its own whole-cluster
test and records NGC 7006's changed role. The old test remains frozen.

## Completed Objective: first pooled cluster RGB experiment

Status: completed as a bounded cluster audit and registered transfer
experiment. The empirical development sequences and once-only whole-cluster
test are documented in `docs/cluster_rgb_calibration_v1.md`; the tested
pooled A/B models are not adopted as a field-star distance product.

Follow the Pro review of `73ad728`. Existing cold-v4/XD, forward-parallax,
warm-giant models and all registered historical test identities stay frozen.
The new label is each cluster member's measured Legacy `g0` minus an
independent whole-system distance modulus, with shared zero-point error.

1. Correct the interpretation of the metal-poor forward model's residual
   scatter and predictive tails, and of the warm giant kernel width. Preserve
   all existing fitted numbers, decisions and products.
2. Recover the complete Vasiliev–Baumgardt M3 and NGC 2419 membership tables
   with their probabilities and Gaia CMD columns. Join every member to actual
   DR11 Legacy grz on allocated NERSC compute. Report identity uniqueness,
   `Pmember>0.9` core, lower-probability diagnostic layer, grz base/clean,
   dust, brightness, position/crowding and usable WISE bands. Spectroscopy is
   diagnostic, not a per-member eligibility condition.
3. Register broad Gaia-CMD phase regions before fitting Legacy sequences.
   Count RGB, HB/RC, AGB, main sequence and ambiguous members separately,
   include independent spectroscopic stage checks where available, and audit
   clean-photometry loss by magnitude, cluster radius, mask and fraction.
   No member selection by model distance or residual.
4. For credible RGB members, construct per-system `(g-r)0`–`M_g` and
   `(r-z)0`–`M_g` Legacy empirical sequences and width summaries. Propagate
   the external modulus and extinction as shared cluster uncertainties;
   contrast M3 low dust with NGC 2419's E(B−V)≈0.06. Compare existing frozen
   model outputs as diagnostics only. Make small inspectable figures/manifests.
5. Use measured RGB number, color/luminosity overlap and label quality to
   decide whether a minimal low-degree cross-system relation is identifiable.
   M3 has 422 clean lower-RGB and only 2 clean upper-RGB candidates; NGC 2419
   has 391 clean upper-RGB candidates and a lower region overlapping red HB.
   Their secure luminosity ranges barely overlap. M53 is preregistered as a
   development bridge and NGC 7006 as a *whole-system reserved test* before
   either system's Legacy RGB photometry is inspected. If a relation is
   identifiable, assess per-system center, scatter and tails. Test
   WISE on the same reliable unblended stars and report its acceptance rate.
   Do not claim photo-only metallicity or phase certainty from two clusters.
   M53 now supplies 156 clean upper and 253 clean lower candidates, bridging
   `M_g≈−0.19` to `+2.73`; NGC 7006 has 19 clean upper candidates reserved.
   The fixed quadratic A/B diagnostic, whole-cluster test and shared-scale
   sensitivity are registered in `docs/cluster_rgb_experiment_protocol_v1.md`.
   The development fit is frozen in `manifests/cluster_rgb_transfer_model_v1.json`:
   model A already has M3/NGC 2419 median offsets +0.688/−0.580 mag.
   It is a transfer diagnostic, not an adopted RGB distance calibration.
   Commit this model before scoring NGC 7006 once; use the holdout to
   quantify the failure or limited domain, not to retune the curve.
   The once-only NGC 7006 test is complete (19 clean upper-RGB stars,
   A median Δμ +0.166 mag, NMAD 0.323 mag, 5/19 |Δμ|>0.5). The small
   holdout and large development-system offsets do not support adopting
   either pooled quadratic. Preserve the cluster loci and report the
   shared distance/dust scale sensitivities; do not retune on NGC 7006.
6. Validate, document in reverse-chronological WORKLOG, commit milestone
   changes and sync code, docs, small manifests and selected plots across
   local, GitHub and NERSC; large catalogs stay on NERSC.

## Next recommended steps

- Add independent low-dust RGB calibrator systems spanning metallicity at
  *overlapping* luminosities, with separate whole-system tests registered
  before fitting. Give their external distance and dust-scale errors a
  shared-cluster treatment.
- Improve phase evidence for red HB/AGB overlap and quantify per-star image
  quality at the bright end. Preserve the existing clean-grz rule until a
  measured recovery layer is validated.
- Only after those systems exist, test a metallicity-marginalized RGB
  luminosity distribution and an explicit RGB/HB/AGB mixture for ordinary
  targets without spectra. Retain the cold-dwarf baseline as a separate
  conditional product.

## Completed Objective: Pro review of caeb218

Keep cold-v4/XD models, routes and earlier final-test identities frozen.
Complete a bounded correction and calibration round with these milestones:

1. Complete: separate general RVS dwarf/giant evidence, cold-dwarf hypothesis conflict,
   4200–5000 K blue-branch training eligibility and calibrated-abundance
   validity. Preserve raw Redrock QSO/galaxy type with a separate quality
   state; the MWS VAC carries `RR_SPECTYPE` and `RR_Z` but no Redrock `ZWARN`
   or `DELTACHI2`. Record CMD white-dwarf evidence independently from the
   spectral label, count conflicts and omit unresolved conflicts from the
   trusted training core. Version the candidate and joint catalogs and
   verify all frozen distance/route columns against earlier products.
2. Complete: review Gaia source 3601173023628559616 by identity, neighbors,
   photometry, image-quality/exposure metadata and available alternate
   measurements; decide its core eligibility without residual-based cuts.
   Extend the joint luminosity plot to include `M_g≈10.19` anchors.
3. Complete: build a targeted giant sample from qualified broad-temperature RVS class
   evidence and actual SpecDis v2.1 / SPDist v1.0 tables. Preserve each
   catalog's identity, quality, distance uncertainty, training correlation
   and applicability; derive Legacy `M_g` from DR11 `g0` and a distance
   distribution. Stratify cool/warm, `logg<1.8`, abundance-valid and
   RGB/RC-ambiguous cases. Audit overlap and label disagreements. The v3
   identity fix yields 2,217/2,044 valid clean SpecDis/SPDist labels;
   both-valid Legacy Mg labels differ by −0.207 mag median. Details:
   `docs/giant_vac_v3.md`.
4. Complete as a bounded, non-adopted model experiment: select independent
   cluster systems with actual
   qualified members and DR11 light; keep whole systems out of fitting.
   Fit and validate a first bounded photo-only giant luminosity relation
   only where label quality and external scale coverage support one.
   The registered grz-only warm evolved-star kernel-mixture test is complete
   (`docs/giant_photo_baseline_v1.md`): 588 reserved stars yield 0.464 mag
   NMAD against SPDist point labels, but five M3 stars show +0.367 mag
   median modulus shift against an independent RR Lyrae reference. Do not
   adopt this model as a giant distance product. A separate
   NGC 2419 RR Lyrae whole-system audit is complete: 13 valid SPDist giants
   have pooled median offset +0.007 mag, while three <5000 K giants have
   median −0.543 mag, so the cool branch needs further scale validation.
   The 4200–5000 K and cooler RGB samples are reported separately. A DESI-first ranking of
   untouched complete DR11 north/south 10° cells identified two southern
   development cells and one separated northern reserved cell. Their DR11
   joint samples have 225/59 conservative 4200–5000 K giants and 173/11
   clean low-dust giants in development/reserved roles. Only 42/2 retain
   valid SPDist labels, so a K-giant model is not fitted from this set;
   `docs/giant_ranked_cool_yield_v1.md` records the selection and label
   bottlenecks. The existing eleven fields supply another 42 SPDist labels
   at 4200–5000 K but do not repair the two-object reserved check. The
   northern reserved optical loss is dominated by bright fraction/mask
   failures, with rejected/retained median `g0` 15.53/18.28.
5. Complete as a bounded spectroscopy-conditioned experiment: combine the seven-field and four targeted-field main-sequence samples
   with source identity deduplication. On development-only data, compare a
   low-dimensional smooth color baseline with and without one continuous
   [Fe/H] term using a forward parallax likelihood for high, medium, weak
   and negative observations. Propagate optical, iron and intrinsic scatter;
   assess parameter/prior sensitivity and reserved-field parallax-space
   predictions. Keep this a feasibility experiment, not a new all-sky release.
   The one-iron model improves reserved parallax log density in five fields;
   β ranges from −0.679 to −0.654 mag/dex in leave-one-development-field-out
   fits. Its use for photo-only targets awaits a metallicity population prior.
6. Complete: validate focused code/data results, update reports and WORKLOG,
   commit coherent milestones and sync programs, manifests and selected
   figures across local, GitHub and NERSC. Large catalogs remain on NERSC
   compute.

## Next recommended science steps

- Keep cold-v4 conditional distances and XD non-adoption decision frozen.
  A global cold-dwarf catalog still needs class probabilities and a
  metallicity-population prior; the spectroscopic iron coefficient alone
  cannot be applied to stars without measured abundance.
- Before another giant fit, identify a low-extinction independent system
  with enough clean K giants and an external distance scale. Audit the
  bright-end fraction/mask losses and SPDist K quality there, using PS1
  only as a photometry cross-check where useful. Preserve RGB, RC and
  subgiant ambiguity explicitly; do not tune to the five M3 members.
- If a photo-only giant experiment is resumed, compare actual grz and
  usable WISE inputs on the same objects, preserve multimodal luminosity
  distributions, and register a new whole-system test before model choice.

## Completed Pro Follow-up Objective

The latest Pro review of `1acf946` is in a bounded joint-calibration round.
Keep cold-v4 models, routes, registered XD adoption gate and historical test
identities frozen. Correct the XD physical interpretation and Gaia point-
reference coverage wording; version the candidate export with existing class
evidence; build a same-star DESI–Gaia–Legacy feasibility sample in seven
registered fields and four DESI-selected DR11 fields. Decide from measured
overlap whether a blue metal-poor
main-sequence relation can be calibrated and identify K giant/cool RGB labels.
No new distance model or deep interval is adopted in this round.

## Current Round Milestones

1. Complete: revise XD result interpretation, label raw interval coverage relative to
   noisy Gaia point references, and correct blue inventory base versus clean,
   low-dust denominators. Keep the original v4 and XD decision records intact.
2. Complete: add per-candidate DESI RVS/Redrock evidence availability, source, quality,
   conservative class and conflict flag. Export versioned v2 FITS products on
   allocated NERSC compute, retaining v1 catalogs and photometric distances.
   Unknown spectra remain unknown. Verify schema and counts field by field.
3. Complete: on allocated compute, assemble a bounded same-star DESI–Gaia–Legacy table
   from `configs/joint_calibration_fields.json`. Start with unique,
   unambiguous DESI matches and report measured/clean grz, low dust,
   4200–5000 K calibrated continuous Fe/H and formal error, conservative
   `logg−sigma_logg>4.2` main-sequence core, transitional/evolved and CMD
   white-dwarf candidates. Retain raw/corrected parallax and weak/negative
   values; derive M_g only for quality high-S/N anchors. Reserve separate
   fields for subsequent validation before any model fitting.
4. Complete: report color, temperature, metallicity, brightness and sky coverage by
   class and parallax layer. Quantify whether the same-star metal-poor core
   supports zero point, color slope and metallicity dependence; otherwise
   state the specific missing overlap. Inventory reliable K giant/cool RGB
   labels and independent luminosity resources for the next science branch.
5. Complete: run focused validation, write concise scientific report and WORKLOG,
   commit milestone(s) and synchronize programs, docs, manifests and selected
   figures through local, GitHub and NERSC. Catalogs remain on NERSC.

Current compute checkpoint: v2 candidate jobs 60484208/209/210 completed
and allocated validation 60484704 confirms all 14 frozen identity/route/
distance columns match v1 for 395,629 objects. The seven-field joint study
60484207 contains 37,214 unique DESI matches, 224 clean low-dust high-S/N
conservative cool-main-sequence stars and only one `[Fe/H]<−1.5` anchor,
none blue of the frozen edge. DESI-first ranking 60484858 identified four
unprocessed DR11 fields with 8/8/6/4 low-dust high-lat metal-poor targets
at *raw* Gaia S/N≥20. Array 60485082 built those fields; dependent job
60485094 found 132,530 unique matches, including 18 high-S/N metal-poor
main-sequence core stars. Five lie blue of the v4 edge (4 development,
1 reserved). The clean metal-poor core has another 42 at S/N 10–20, 228
at S/N 0–10 and 3 negative corrected parallaxes. One blue high-S/N object
has discordant optical and Gaia/spectral colors. Targeted fields include
230 conservative cool giants, 103 clean low-dust, but zero direct high-S/N
geometric giant anchors. No new model is adopted.
The cold-v4 and registered XD decisions remain frozen.

## New Round Milestones

1. Complete: reconcile the prior review's interpretation and publish an explicit
   development/selection protocol and candidate-output schema.
2. Complete: export v4 per-candidate A/B/C choice, higher-tier fallback reasons, support
   ratios, infrared-conflict marker, cold-dwarf-hypothesis status and validated
   depth state without changing distances or deleting conflicts.
3. Complete: build a full-covariance heteroscedastic deconvolved Gaussian mixture
   `p(M_g, colors)` on exactly the v4 A/B/C training identities. Use 16/32/64
   components and propagate shared-band and Gaia label uncertainty.
4. Complete: compare configurations by held-out development sky blocks. On identical
   accepted objects, compare chosen candidate against frozen v4 in bias,
   NMAD, tails, probability quality, g/r perturbation response and error/color
   strata. The registered gate rejected XD K=32; no new final test follows.
   A separate color-stratified g/r response diagnostic confirms the same
   direction in both occupied color bins without altering that decision.
5. Complete: define the metal-poor blue branch from reliable Gaia/DESI labels: coverage,
   color-boundary failures, CMD position and giant/nonstar confusion. Keep it
   distinct from the frozen cold-v4 box. The current four-field inventory
   and label/anchor contract are in `docs/cold_v5_blue_branch.md`.
6. Complete: validate, document, commit and synchronize programs, small manifests and
   figures across local, GitHub and NERSC. Keep catalogs and model files on
   NERSC and use allocated CPUs for scans/training. Three candidate exports
   and single-table checks are complete; 81 tests pass. The final color
   response manifest and figure close the comparison record.

## Completed Previous Round Milestones

1. Complete: recompute common-acceptance crossover centers, paired differences,
   directly bootstrapped median differences and common-sample NMAD from the
   existing per-source arrays. Keep full-flow acceptance and NMAD visible.
2. Complete: correct the weak-parallax subgroup PIT for each observed-SNR truncation
   interval; simulate the same selected statistic to calibrate its center.
3. Complete: record each tier fallback reason and complete WISE quality evidence.
   Compare a conflict marker and a conservative hold policy on development
   dwarf, giant, metal-poor and unlabeled candidate samples. Assess boundary
   stability under reported measurement errors.
4. Complete: force A/B/C on identical supported stars in the three native fields;
   report Gaia residuals and within-star tier differences. Match same-tier
   complete inputs and photometric quality across fields; separate label and
   model-prediction contributions, with parallax/brightness strata.
5. Complete: measure fixed-tier PS1 g/r/z/grz substitutions and ±0.005/0.01/0.02 mag
   band responses on the same qualified stars, plus optional rerouting.
   Report center, scatter, tails, support-edge and color dependence.
6. Decision: retain v4 distance routing and add a diagnostic infrared support
   conflict/type-risk flag. A hard rejection loses 3.5%–4.1% of strict Gaia
   cold anchors and 13.4%–14.7% of quality-qualified DESI cold dwarfs in the
   two useful development fields. A future hard-hold or fitted correction
   needs separate calibration and independent final validation.
7. Complete: validate code/results, update reports/worklog, and sync committed
   programs, manifests and selected figures to local, GitHub and NERSC.

## Previous Diagnostic Round (through 88f7330)

The preceding Pro-review requirements are complete: the frozen model and route
were held fixed; extinction algebra, selection-aware weak-parallax PIT,
WISE20 wide-binary and DESI boundary checks, and the same-source 2×2
noise/template crossover are documented in `docs/pro_latest_round.md`.

Three preregistered independent native-transfer fields have been scored once.
South RA330 and RA20 medians are −0.0178 and −0.0520 mag; north RA130 is
+0.0687 mag. The contrast persists at corrected parallax ≥2 mas, after
outcome-blind source matching, in RUWE≤1.1 and ordinary tangential-speed
strata. Six preregistered sky-census fields repeat positive medians in
three informative positive-b regions; four opposite-sign Galactic/ecliptic
fields yield too few low-extinction anchors to isolate coordinates.

The bounded foreground-dust sensitivity job 60462368 is complete. Scaling
the catalog extinction correction to 75% leaves north-minus-south contrasts
within 0.003–0.004 mag of baseline; the difference persists even with no
correction. See `docs/cold_v4_dust_sensitivity.md`. This is explanatory use
of already inspected fields, not a fitted dust map.

The PS1 relative grz audit was registered before reading PS1 rows and is
complete. On the restricted PS1-quality samples, Legacy-minus-transformed-
PS1 g medians are +0.0089/+0.0046/−0.0039 mag in south RA330/south RA20/
north RA130. Same-source three-band substitution changes the frozen route's
residual centers, but the restricted color and footprint coverage plus
Legacy/PS1 calibration ancestry make this a sensitivity diagnostic. See
`docs/cold_v4_ps1_relative_audit.md`.

## Outstanding Issues and Validation Criteria

- A causal optical-system test needs an independent reference with a
  validated cool-star bandpass transformation or a matched cross-field
  training/calibration design. The existing PS1 comparison constrains
  relative measurement response on only its qualified subset.
- Preserve the v4 forests, color box, support and WISE20 policy. The current
  median distance remains conditional on the cold-dwarf hypothesis. Deep
  nominal intervals are still withheld because the independent deep test
  undercovers in multiple measured-r bins.
- If PS1 or a later independent source identifies a plausible cause, register
  new calibration and final sky regions before fitting a correction. Existing
  final regions stay explanatory and cannot validate that correction.
- Run targeted tests, verify frozen hashes and source identities, inspect
  figures, update documentation and sync committed programs, reports, small
  manifests and figures across local, GitHub and NERSC. Catalogs and models
  remain on NERSC.

## Milestones

1. Complete: freeze baseline commit, model checksums, original test predictions and
   nside=64 test cells. Preassign one external calibration and two external
   final-test fields before examining model residuals.
2. Complete: reconcile strict and recovery nonoptical conditions; require DESI quality
   and calibration domain in spectral residual bins. Record impact on counts.
3. Complete: decompose the recovered bright-star offset with matched strict controls,
   out-of-fit strict predictions and paired Legacy/PS1 photometry replacement.
4. Complete: run the three external fields with unchanged selection and first measure
   direct transfer of the frozen baseline models.
5. Complete: calibrate 68%/95% intervals on the preassigned external calibration field;
   score them once in the two final-test fields, including red-color bins.
6. Complete: run raw-flux noise injection at observed target-depth S/N and report
   acceptance, residual and interval coverage by layer and depth.
7. Complete: quantify target input coverage and pure-photometry cold-branch selection
   against independent Gaia/DESI labels. Plot, document and sync all results.
8. Complete: correct the paired PS1 statistics in the report, and rerun
   target coverage with a training threshold defined by the tenth *other*
   object (k=11 self-query; k=10 external query). Publish clearly named
   color-box recall, color-plus-support recall, and known-nonstar leakage.
9. Complete: before any source inspection, freeze `configs/next_fields.json`. Its
   RA 205–215/Dec −10–0 field is development calibration; RA 230–240/Dec
   5–15 and RA 50–60/Dec −30–−20 are untouched final tests. The previous
   north/south fields now count as development evidence for any new method.
   After the first final field yielded only 25 cold anchors due the fixed
   E(B−V) cut, register two additional high-latitude fields in
   `configs/supplemental_fields.json` before reading their source rows.
   Evaluate and report all four final fields regardless of their counts.
10. Complete: build a revised noise experiment on the identical anchor stars, identical
    target noise rows and one shared five-band realization for A/B/C. Draw
    noise from a grz mother pool including weak WISE detections; quantify
    source-template noise, all-tier acceptance, strict quality, Mg and final
    distance-modulus/distance residuals and interval coverage. Cluster repeat
    realizations by source for uncertainty.
11. Complete: fit bias correction and conditional 68/95% interval expansion using only
    application-visible Legacy/WISE colors, magnitude and photometric errors.
    Keep fit, calibration and final test identities disjoint. Report bias,
    NMAD, both coverage rates, low/high misses, interval width, count and
    spatial-block uncertainty by color and depth. Compare same-star A/B/C
    under a common training identity set before attributing band benefit. The
    original cold_baseline.py already supplies that common-fit comparison;
    retain its result while assessing new external-field performance.
12. Complete: audit dwarf, giant, nonstar and other/unknown labels across color,
    brightness and sky; state selection of each labeled sample. Evaluate
    class-conditional recall and leakage. Use cold-dwarf-hypothesis distances
    with type-risk and support flags until calibrated type probabilities are
    possible. Keep APOGEE auxiliary and recovery candidates separate.
13. Complete: commit and synchronize programs, manifests, reports and selected figures
    among local, GitHub and NERSC; keep catalogs and models on NERSC. Review
    final-field metrics and unresolved limits with the user.
14. Complete: export a bounded photometry-only candidate demonstration in
    an independent field, with frozen-model median distances and explicit
    support, optical quality, depth and unverified-type flags. No class
    probability or calibrated interval is assigned.
15. Complete: audit the fixed highest-available A/B/C routing rule together
    with color-box and support selection on all four registered final fields.
    Report the selected fraction and direct frozen-model distance performance,
    including Gaia-color diagnostics, before any claim about a selected catalog.
16. Complete: register cold_v4 expanded strict-anchor training, an independent
    route-calibration field and three new untouched 10° final fields before
    reading their source rows. Fit the tier-specific forests, add feature-
    matched support and calibrate the complete routing flow without a center
    correction. Evaluate every registered final field and keep the original
    c3637cb forests/results frozen. All three new final fields completed on
    NERSC; the fixed route selects 4,991/4,784/2,681 strict cold anchors
    and achieves final-modulus 95% coverage 96.0%/94.7%/95.0%.
17. Complete: pass the shared five-band flux-noise experiment through the
    complete cold_v4 color box, feature-matched support, tier fallback,
    forest and route-calibrated intervals. Report acceptance, strict quality,
    final distance-modulus coverage and source-cluster uncertainty by depth.
    The four r=18–22 bins have 95% final-modulus coverage
    92.5%/91.8%/92.4%/91.0%; the deepest bin's strict-quality fraction is
    32.3%, and W2 is selected in only 51/1,784 draws. The observed failure
    is a depth-calibration limit, not a reason to change frozen final tests.
18. Complete: without changing forests, report A/B/C × clean/strict/other
    image-quality and error/support strata for calibration, new final Gaia
    fields, shared noise and real photometric candidates. Include per-tier
    bias/NMAD/68%/95% coverage/tail misses/width, sky- or source-cluster
    uncertainty, and complete-route intersection with grz_clean. The real
    candidate/Gaia audit and new deep source-cluster tier diagnostics are
    published; sparse deep C is explicitly recorded.
19. Complete: applied the complete v4 route to deduplicated, quality-qualified
    Gaia/DESI labels in the new fields. Measure dwarf recall and giant/nonstar
    leakage by tier, clean state, fallback, color, magnitude, errors and sky;
    separately audited calibrated-domain metal-poor labels. Two final fields
    have DESI matches; RA0–10 has none. Defer type probabilities until label
    selection and class priors are supportable.
20. Complete: corrected simulated flux IVAR when template noise exceeds
    target noise, and corrected the report's description of borrowed masks.
    Compare only a few predeclared routing policies on identical five-band
    development observations, considering precision, coverage, width and
    acceptance. Development chose and hash-locked `grz_clean` plus WISE
    S/N>=20; tested it on separately registered fields.
21. Complete: assembled high-S/N templates and WISE-unconditioned target-noise
    environments from several fields; isolate fit/development, calibration
    and final-test identities and sky blocks. Calibrate final distance-modulus
    intervals by chosen tier and measurable noise/support using independent
    sources. RA35–45 calibration completed with one observation per source
    per target-noise depth. Superseded v1 grouping by injected target-noise r;
    v2 groups by measured r and deduplicates source identities in each bin.
    The one-time RA170–180 final test has 95% coverage 92.6%/92.9%/95.1%/92.3%
    and 68% coverage 62.6%–65.2% across measured r18–22. Deep v2 intervals
    remain diagnostic pending a new development/calibration/final sequence.
22. Complete for current scope: used isolated wide-binary/cluster reference systems for color- and
    brightness-dependent distance checks, then assess weak/negative Gaia
    parallaxes with a parallax-space forward model and documented selection.
    Examine metal-poor, binary, extinction and field dependence of bias.
    Initial final-field check has 456 clean-routed pairs and 201 Blanco 1
    members; the pair fainter-minus-brighter median is +0.046 mag.
23. Complete: publish the bounded application-range report, selected figures,
    small manifests and validation; update README/WORKLOG/PLAN, commit and
    sync local, public GitHub and NERSC. Keep catalogs/models on NERSC.

## Scope

Original field RA [0,10), Dec [-10,0) remains a locked baseline. Prior field
roles and source/model hashes are in configs/followup_fields.json; new roles
are in configs/next_fields.json. The strict optical and Gaia criteria remain
the comparison standard. Cold-v4 roles and fixed method are in
configs/cold_v4_fields.json. The full-footprint scan remains stopped.

## Next Recommended Objective

Keep frozen v4 as the conditional cold-dwarf distance baseline. Build a
targeted K-giant/cool-RGB class and luminosity sample from qualified DESI
spectra, nearby high-S/N Gaia giants, clusters and independently audited
spectroscopic-distance resources. Register development and final identities
before fitting or evaluating a new giant relation. Keep class and metallicity
uncertainty explicit for no-spectrum application targets.

For the metal-poor main-sequence extension, the targeted DR11 sample supplies
18 high-S/N anchors below `[Fe/H]=−1.5`, five blue of the old edge and only
one blue in reserved validation. Audit the discordant-color case and seek
more clean geometric anchors before fitting a blue color–luminosity relation.
Develop a forward-parallax likelihood for the 42 intermediate, 228 weak and
three negative-parallax clean metal-poor core objects; preserve the spectral
selection model and source identities. The XD comparison remains a research
candidate under its registered gate, with no change to v4 output.
