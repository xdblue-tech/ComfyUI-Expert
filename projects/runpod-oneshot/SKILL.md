---
name: runpod-oneshot
description: Compile exactly one positive and one negative NSFW image prompt for the unholydesiredmix checkpoint (Illustrious-family SDXL, tag dialect). Prompt output only — no generation, no scripts, no workflow files. Use when the user says runpod-oneshot, or asks for prompt compilation for unholydesiredmix / unholy-sdxl / salisse without launching a render.
---

# runpod-oneshot — unholydesiredmix prompt compiler

One job: turn the user's description into **one positive prompt and one negative
prompt**, both copy-paste ready, both written for the unholydesiredmix
checkpoint. Do not generate images. Do not call APIs. Do not run scripts. This
file is the entire skill — it ships alone, with no bundled files.

unholydesiredmix is an **Illustrious-family SDXL checkpoint** running as
txt2img on the unholy-sdxl workflow (steps 30, cfg 7, `dpmpp_2m` + `karras`,
~1M pixels). Everything below is scoped to that one target:

- Tag prompts: comma-separated danbooru-style tags, 40–120 tags. Never English
  prose paragraphs, never Flux/Pony cross-dialects.
- Never paste `score_*` Pony headers.
- The positive always starts with the fixed salisse quality header below — it
  replaces the generic Illustrious header.
- CFG / steps / resolution live in 参数旁注 only, never inside the positive.
- Positive and negative are always emitted as two separate blocks (the
  unholy-sdxl workflow has a dedicated negative field, node 3).
- Censor nouns (`censored`, `mosaic`, `bar censor`) appear in the negative
  only — never in the positive.

## Step 1 — Parse into the IR

Fill this internally; do not print it unless asked for a slot dump.

| Field | Required | Default (fills unspecified slots only) |
|-------|----------|---------|
| subject.gender_count | yes | `1girl, solo` |
| subject.age | yes | `18 years old` |
| subject.ethnicity | no | `japanese` |
| subject.body | no | `perky G-cup, hourglass figure, slim waist` |
| subject.face | no | `almond eyes, beautiful detailed face` |
| nudity.level | yes | L2; 全裸/脱光 asked → L5; 展示/张开/特写 → L6 (user override) |
| clothing.state | yes | L2: robe worn, one shoulder slipped, hips covered |
| anatomy.female_explicit | if L5–L6 or 私处 asked | true |
| anatomy.male_explicit | if male genitals on-screen | false |
| pose.body | yes | `standing, three-quarter view` |
| pose.hands | yes | limb-lock (below) |
| pose.legs | yes | `legs slightly apart` / `thighs together` |
| expression | no | `sweet expressive gaze` (one expression) |
| scene | no | night garden (stone lanterns, cherry blossoms) |
| camera.shot / angle | yes | `medium shot` / `eye level` |
| lighting | no | `Rembrandt side light` + 1 modifier max |
| lora_triggers | no | user list, verbatim, front of body tags |
| multiperson | if 2+ | exact count + `no extra people` + per-person tags |

### Nudity levels (clothing changes; subject, scene grammar stay)

- **L1** clothed silhouette — robe worn closed, tied.
- **L2** 1–2 leaks (default) — one shoulder slipped, cleavage, hips covered.
- **L3** half-nude — topless with hips still wrapped, or bottomless with chest covered.
- **L4** scraps / high exposure — half-fallen garment still worn, breasts out, lower body readable.
- **L5** fully nude — no worn clothes: `fully nude, full frontal nudity, nipples fully visible and unobstructed`.
- **L6** display (user override of L5, not a seventh rung) — `legs spread wide open, pussy fully displayed, presenting genitals to camera`.

### Conflict table (fix before rendering)

| A | B | Fix |
|---|---|-----|
| `pussy` visible | intact panties | `no panties, pussy visible` |
| `no_cover` | `hand between thighs` / `arm across breasts` | hands to prop or hanging at side; never covering |
| sheer fabric as the only reveal | nudity asked | `fully nude` + `fabric not covering breasts or genitals` |
| Illustrious tags | long English paragraphs | re-render as tags |
| bare `pussy` | explicit genitals asked | expand via anatomy tags (Step 2) |
| phone / CCTV device | `8k, masterpiece` quality soup | match device quality |

### Phrase map

| User says | IR |
|-----------|-----|
| 优化提示词 | edit mode (below), preserve style |
| 标明隐私部位 / 生成隐私部位 | `explicit_genitals=true` + anatomy tag line |
| 不要遮挡 / 无遮挡 | `no_cover=true`, L5 (L6 only if they asked 张开/特写) |
| 全裸 / 脱光 | L5 |
| 张开 / 特写私处 | L6 + closeup |
| 半裸 / 解开衬衫 | L3, keep one garment |
| 不要全裸但私处可见 | L3/L4 + `no panties, pussy visible` |
| 和风 / 庭园 / 扇子 | keep garden + fan defaults |
| 手不要挡 | pose.hands = raised or prop, never crotch |
| 温泉 / 泡澡 / 湿身 | onsen scene (towel discarded at feet, wet skin) |
| 加班 / OL / 办公室 | after-hours office scene |
| 夜店 / 霓虹 | nightclub booth |
| 纹身 | styling.tattoo + fusion words mandatory |
| 道具：扇子 / 毛巾 / 酒杯 / 冰块 | style.props; never cover under `no_cover` |
| 胶片感 / portra / 黑白 | map film/lighting tags |
| 双人 / 百合 / 男女 | gender_count update; each person own pose line |
| 更高 / G杯 / 御姐 | keep house subject (already G-cup / almond eyes) |

Non-trigger: 画一张 / 生成图片 / 出图 without asking for a prompt — compile
and stop; never call an image API from this skill.

### Edit mode — optimizing a pasted prompt

When the user pastes a long existing prompt:

1. Extract slots that already exist (subject, scene, light, hair).
2. Apply their constraints (`隐私部位`, `不要遮挡`) as IR overrides.
3. Delete covering fragments (`hand between thighs`) instead of stacking
   anatomy on top of a cover.
4. Re-render fully in the tag dialect — do not keep contradictory fragments.

## Step 2 — Render the positive

Fixed quality header, verbatim, before all compiled content. Do not drop,
reorder, or reweight entries. The two `<lora:>` tags are part of the header
contract — keep them even though unholy-sdxl parses them as literal tokens
(LoRA loading is handled worker-side or not at all; that is out of scope here).

```text
masterpiece,best quality,amazing quality,very aesthetic,absurdres,ultra detailed face,ultra detailed eyes, detailed, depth of field, high detail, best quality, very aesthetic, 8k, volumetric lighting, cinematic, dynamic shadows, depth of field, <lora:StS-Illustrious-Detail-Slider-v1.0.safetensors:0.3>, <lora:Quality_Modifier_Illust.safetensors:0.5>,
```

Body tag order after the header:

```
{header},
{user LoRA triggers, verbatim},
{count tags: 1girl, solo | 2girls, lesbian | 1boy, 1girl},
{character: 18 years old, japanese, long wavy hair, almond eyes, ...},
{body type tags (1–2): perky G-cup, slim waist, wide hips, thick thighs},
{nudity-level tags (per L above)},
{anatomy tags (if explicit)},
{pose tags: standing, three-quarter view, limb-lock},
{expression (exactly one from the lexicon)},
{skin/hair/ornament tags (2–4)},
{scene tags}, {one lighting main + one modifier},
{camera: shot + angle}
```

Weighted syntax works: `(term:1.3)`. Underscore tags preferred. Keep the whole
positive within SDXL's token budget — trim scene depth before subject detail.

### Expression lexicon (pick exactly 1 per render)

| Mood | Tags |
| --- | --- |
| Shy | `shy expression, looking away, blushing cheeks, covering mouth with fan` |
| Seductive | `seductive gaze, half-lidded eyes, biting lip, smirking` |
| Pleasure | `eyes half closed, mouth open, moaning, ahegao, flushed face, tears of pleasure` |
| Submissive | `looking up through lashes, pleading eyes, parted lips` |
| Melancholic | `lonely yet elegant, wistful gaze, distant expression` |
| Playful | `playful smile, tongue out, winking, laughing` |
| Sadistic | `sadistic smile, smirk, looking down at` (pairs with femdom block) |

- One set only — mixing two merges the face into a neutral mask.
- `ahegao` belongs to Pleasure; never combine with Shy or Melancholic.
- User-named moods (冷淡/嗔/媚) map to the closest set; never silently
  swap back to sweet.

### Anatomy tags (never emit bare `pussy` alone when explicit is asked)

```
uncensored, pussy, pussy juice, labia, clitoris, nipples, areolae,
completely nude, no panties
```

The fixed header already carries `nsfw, explicit, uncensored`; do not duplicate.
Pubic hair, pick 0–1: `shaved smooth, neatly trimmed landing strip, natural bush`.
Breasts focus: `puffy areolae, erect nipples, inverted nipples, lactation (only if asked)`.
Male: `penis, testicles, abs, navel` + pose; erect vs flaccid must be stated.
Fluids, minimal quantifiers only: `single drop of sweat, dewy sheen, thin streak of arousal wetness` — never `covered in / drenched` unless asked.

### Limb-lock (default when no pose specified)

Prose rule: right hand holds a closed fan beside the right cheek, left arm
hangs empty, both feet planted, exact counts stated. Tag rendering:

```
holding closed fan near own cheek, left arm hanging at side,
two hands, two arms, two feet, standing
```

If the user specified a pose (举手 / 叉腰 / 坐 / 跪 …), keep their pose and only
add counts + left/right so limbs do not scramble.

### Multi-person (2+)

Exact count first (`2girls`), spelled out with composition: 
`Exactly five people — four 22-year-old adult Japanese beauties and one handsome 25-year-old Japanese man`., 
then `no extra people`. One tag group per person, each attribute (hair color, 
garment color, hands, expression, L-level state) inside its owner's group; 
never a shared-attribute tag that could migrate between people. Every 
prominent hand is possessive and anchored:
`girl one's right hand on girl two's shoulder`, never `hand on shoulder`.

Directional interactions: Subject–verb–object with named targets: `the woman in crimson kisses the man`, 
`she glares past his shoulder at the woman in crimson` — never `they kiss` or `the women glare at each other`.
Rivalry/jealousy webs (multiple named glances between women) validated in R28; 
keep one glance target per person to avoid stacked head rotations.

Identity tokens: Double-bind each person: hair color + robe/garment color 
(`raven-black for the woman in crimson`), stated once per person in her own sentence. Shared styling 
(ornaments, makeup) may stay in one global sentence — that behaves like a validated common-prompt block.

Camera and framing: One simple camera clause only; complex modifiers (from above, dutch angle) 
destabilize 3+ person scenes. Pull back for 4+: 
`medium-wide shot, 50mm lens, camera pulled back so all five figures fit fully in the wide frame`; 
wide/landscape framing helps. Keep the background explicitly clean: `clean and uncluttered`.

Spatial staging: Give each person a distinct body level rather than a same-height cluster: 
standing / seated / kneeling behind / reclining / straddling all passed in R28. 
Anchor positions to the viewer: `at the viewer's right of the man`, `reclines along the lower basin edge at the viewer's left`.
A named central anchor (man seated on a fountain rim) plus per-person relation to it keeps 
5 people from merging. Straddling + a literary knee-line fabric phrasing passed as-is (R28); 
if the wrap and the partner's waist fabric merge into a censor patch, downgrade contact first: 
`straddles his lap` → `kneels upright across his lap, her weight held above him`.

Identity tokens: Double-bind each person: hair color + robe/garment color 
(`raven-black for the woman in crimson`), stated once per person in her own sentence. 

### Recipes — Illustrious tag seeds

These seeds are rendered in the Illustrious dialect, not Pony: no `score_9` /
`score_8_up` / `score_7_up`, no `rating_explicit`, no `masterpiece, best
quality, absurdres` tail. The fixed salisse header stands in for the stock
Illustrious header (`masterpiece, best quality, newest, absurdres, nsfw,
explicit, uncensored`). Body tags below the header are otherwise identical to
the Pony body grammar (character, nude, anatomy, pose, scene). Use as IR
seeds; re-render, do not paste blindly.

**R1 和风全裸无遮挡** (default beauty, L5) — body block after the fixed header:

```
1girl, solo, 20 years old, japanese, beautiful face, almond eyes, glossy lips,
long wavy hair, half updo, hair ornament, jeweled hairpiece, tassel earrings,
completely nude, full frontal, nipples, areolae, large breasts, perky breasts,
pussy, pussy juice, labia, clitoris, uncensored pussy, no panties,
legs slightly apart, standing, three-quarter view, holding fan,
covering own mouth, sheer black fabric around hips only, japanese garden,
cherry blossoms, stone lantern, torii, night, petals, rembrandt lighting,
cinematic lighting, bokeh, film grain
```

**R2 白衬衫半裸、私处仍可见** (L3/L4): `oversized white shirt unbuttoned,
bare breasts, nipples visible, shirt hem not covering, no panties, pussy
visible, sitting on windowsill` — never keep panties.

**R3 私处特写** (L6): `extreme close-up, crotch-level, 85mm macro` + full
anatomy tag line + `labia spread open showing pink interior`.

**R4 成人制服教室** (L3): `sailor
shirt unbuttoned, one breast exposed, skirt hiked up, no panties, pussy
visible, sitting on desk`. 

**R5 浴室蒸汽** (L5): `wet skin, droplets, steam, fully nude, nipples,
explicit vulva, no covering hands`. Phone device allowed — then add no
extra cinematic/studio tags beyond the fixed header.

**R6 监控更衣室** (voyeur): `CCTV footage, grainy, timestamp overlay,
voyeur angle, door crack` — surveillance quality, never cinematic grading;
the fixed header stays (workflow contract) but add no extra quality words.

**R7 双人**: `1boy, 1girl` or `2girls`; each body gets its own pose line;
never let the second person become a covering prop unless asked
(conflicts with `no_cover`).

### Femdom / BDSM tags (on request only)

Not part of any default recipe. Only when the user asks for femdom, dominatrix,
bondage, or pegging. Mix one row per group into the standard body; the fixed
header stays first, anatomy block stays in place.

| Group | Tags |
| --- | --- |
| Outfit | `dominatrix outfit, black latex, thigh-high boots, elbow gloves, corset, choker, fishnet stockings` |
| Gear | `riding crop, leather whip, collar and leash, ball gag, blindfold, rope, shibari, handcuffs` |
| Power frame | `femdom, dominant woman, submissive man, bound, restrained, kneeling man` |
| Pegging | `strap-on dildo, harness, pegging, anal, from behind, female on top, female penetrating male` |
| Pose/expression | `standing over, foot on chest, looking down at, sadistic smile, smirk, parted lips, half-closed eyes` |

Cautions:
- Pick ONE act frame (pegging OR facesitting OR edging) — stacking act verbs
  merges bodies.
- Gear tags bleed: `rope` + `shibari` together often draws straps across
  untagged zones; use one or the other.
- Pegging needs the man's pose explicit (`all fours, grabbed hips`), or the
  model renders a standing tangle.

### Scene seeds

One scene + optional theme. Default beauty scene: dark Japanese garden night.

| Scene | Rec level | Taboo |
|-------|-----------|-------|
| bedroom morning | L2–L4 | none |
| rainy window living room | L3–L5 | dry clothes contradiction |
| bathroom / shower | L5–L6 | covering hands, steam as censor |
| open-air onsen night | L5 | towel clutched to front |
| after-hours office | L3–L4 | full nude standing at desk (illogical) |
| classroom after school | L3–L4 | — |
| rooftop sunset | L2–L4 | — |
| library corner | L3 | loud setting mismatch |
| train sleeper / last subway | L3–L5 | crowd logic errors |
| car backseat night | L4–L5 | seatbelt clipping anatomy |
| fitting room mirror | L2–L4 | — |
| convenience store night shift | L2–L3 | counter blocking view |
| rooftop neon night | L4–L5 | — |
| beach sunset | L4–L5 | tan lines are a plus, not censor |
| forest clearing daylight | L5 | modern indoor props |
| shrine / festival yukata | L2–L4 | sash fully removed breaks outfit identity |
| nightclub booth | L4–L6 | sober-clean look (sweat/sheen fits) |
| love hotel pink room | L5–L6 | — |
| tattoo studio | L3–L5 | sticker-look tattoos (use skin fusion) |

Theme seeds (drama): cheating-wife drama, OL
overtime loneliness, hot-spring trip with lover, idol off-duty, exhibitionist
thrill outdoors, voyeur caught changing, post-gym shower, maid
service fantasy, first night after wedding,
reunion after long trip. Non-consensual premises stay refused.

Scene quality coupling (tag dialect): onsen/bath → wet skin + steam, towel
discarded never clutched; nightclub → colored gels on skin, sweat sheen,
halation; phone selfie → casual framing, no studio words beyond the fixed
header; CCTV → surveillance grain + timestamp, never cinematic grading.

### Pose library

Body + hands + legs are three separate slots. Hands decide whether
`no_cover` survives.

Standing: `standing, weight on one hip` / `standing, back arched against
wall` / `standing, one foot propped on tub edge` / `stretching arms overhead
(torso lifted, not covering)` / `turning to look back over shoulder`.

Sitting: `sitting on desk edge, legs dangling` / `sitting on windowsill, knees
drawn up but open below` / `straddling a chair backwards` / `floor seat,
hugging one knee` / `sofa lounge, legs tucked to the side` / `leaning back on
both hands behind hips`.

Kneeling: `seiza, hands resting openly on thighs` / `kneeling upright, chest
out` / `all fours, back swayed` / `kneeling between partner's legs`.

Lying: `lying on back, knees bent, feet planted` / `lying on side, top leg
drawn up` / `on stomach, calves raised and crossed` / `legs over head
(display)`.

Display (L6): `legs spread wide open` / `squatting, heels together` /
`presenting on all fours, looking back` / `missionary spread toward camera`.

Legs: `slightly parted` / `one leg raised onto the tub edge` / `crossed at
ankles but open at knees` / `spread wide open (L6)` / `wrapped around
partner` / `over partner's shoulder`.

Hands catalog — safe under `no_cover`:

```
one hand holding a folding fan half-covering her face
other hand raised near her head, fingers in her hair
both hands raised beside her head against the wall
hand gripping the lifted skirt hem at her waist
loosely holding reading glasses / cocktail / phone
arm draped over sofa backrest, away from her body
fingers laced behind her neck
palm flat on the mirror beside her face
one thumb hooked in waistband pulled low
hands on own collarbones, elbows out
```

Ban list under `no_cover`: `hand between thighs`, `covering crotch`,
`covering breasts`, `arm across chest`, `hugging knees over chest`,
`towel clutched to front`. Cover-failure drama only if asked:
`hand trying to cover but too small`.

Multi-person: each person needs their own pose line; positions explicit
(`kneeling between his legs`, `her back against his chest`), never a vague
`embrace`; for 2girls, name who does what.

### Styling

Pick 2–4 items total across styling; precision beats pile-up.

Makeup: `dewy natural base` / `soft matte base` / `long lashes, subtle
aegyo-sal` / eyeshadow `champagne`, `rose brown`, `smoky charcoal` /
`glossy pink lips`, `matte nude`, `smudged red lipstick (post-oral drama)` /
`blush on cheek apples` / nails `short nude gloss`, `red almond` / `tiny
sparkling flower sequins near eye corners`.

Hair: `long flowing wavy, half-tied loose bun with stray strands` / `straight
black with hime cut sides` / `high ponytail` / `messy bun` / `wet hair slicked
back from bath` / `twin tails` / `bob with inward curl`.

Body realism (2–3 max): `faint freckles over nose bridge, beauty mark under
left eye` / `subtle tan lines from swimwear` / `goosebumps trails` /
`defined collarbones, soft hip dips` / `dewy sheen, visible skin pores`.

Tattoos — four-layer fusion mandatory: pattern + body anchor + ink color +
fusion words; without fusion words tattoos render as stickers. Patterns:
`small heart (wrist)`, `peony spray (thigh)`, `koi and waves (back)`, `snake
coiling (arm)`, `cherry blossom branch (collarbone)`, `kanji calligraphy
(nape / hip)`, `band of lace script (ribs)`. Ink: `black linework`,
`red-black gradient`, `full-color traditional`. Fusion words (always 1–2):
`ink settled into skin pores`, `slightly sunken under skin texture`,
`natural skin tone showing through the black`, `matte finish not glossy`,
`edges softly blurred beneath the skin surface`.

Props (one scene-changer each): `black folding fan half-covering her face` /
`white towel discarded at bare feet (onsen — never clutched to front)` /
`wine glass with condensation` / `ice cube traced along collarbone` / `phone
in mirror selfie pose` / `leather leash slack between hand and thin choker` /
`cat-ear headband + bell choker` / `reading glasses held
loosely`. Props never cover breasts or genitals when `no_cover=true`.

## Step 3 — Render the negative

Always emit a negative. Illustrious / SDXL-anime standard base block:

```
low quality, worst quality, normal quality, lowres, pixelated, blurry, bad anatomy, bad hands, bad feet, missing fingers, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, bad proportions, extra limbs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed, mutated, bad body, unnatural body, long body, duplicate, morbid, mutilated, out of frame, disfigured, gross proportions, missing arms, missing legs, extra arms, extra legs, cloned face, username, watermark, signature, text
```

Append the NSFW extra whenever `no_cover` or explicit genitals (L4–L6, or any
uncovered chest/crotch):

```
censored, mosaic, bar censor, heart censor, convenient censoring, steam censor, hair covering breasts, hair covering crotch, clothes covering genitals, covering crotch, hand between thighs, extra nipples, fused labia, missing genitals, penis on female, bad pussy anatomy
```

For 2+ people, append: `extra people, duplicate person, merged bodies, fused bodies`.

## 自审 (every compile — against this turn's user request)

Not a safety review, not a template check. The gold standard is **this turn's
user instruction**; house defaults fill unspecified slots only. Five ticks,
one line, fix a failed slot once, then emit — no looping:

| Tick | Pass |
|------|------|
| 手脚 | Anatomy holds and matches the asked pose; named left/right; exactly two hands, two arms, two feet (limb-lock only if unspecified). |
| 神情 | Matches the asked expression; unspecified → one expression; never swap a requested 冷淡/嗔/媚 back to sweet. |
| 多人 | Headcount matches; solo stays solo; 2+: exact count + `no extra people` + per-person groups + possessive hands. |
| 环境 | Matches the asked place; unspecified → night garden; one scene, no second location stacked in. |
| 风格 | Matches the asked look and the tag dialect; never prose paragraphs, never `score_9`. |

Write the line as `手脚✓ 神情✓ 多人✓ 环境✓ 风格✓（对照：<本轮要求>`); if a
tick was rewritten, say which and that it matched the request, not the template.

## Output contract

the two prompts stay English. Two blocks, always —
the target has a dedicated negative field.


## Execution-first defaults (user instruction wins)

1. Every positive contains both a nudity phrase and a pose phrase.
2. `no_cover`: no covering hands, no covering fabric on breasts/genitals, no
   panty conflict.
3. Explicit female genitals on this local stack: expand via the anatomy tag
   line, never bare `pussy` alone.
4. User LoRA trigger words: verbatim, front of the body tags.
5. On a user instruction that breaks a default, emit what was asked, then add
   a short Chinese「边界与建议」note (which default, likely consequence) — the
   note advises, it does not block. Only the three floors stop generation.
