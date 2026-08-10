อันนี้คือ **SwiGLU-style feedforward MLP** — ส่วนที่อยู่หลัง attention ในแต่ละ decoder layer ทำหน้าที่ "คิด/ประมวลผล" ข้อมูลเชิงลึกในแต่ละ token แยกกัน (ต่างจาก attention ที่ให้ token คุยกันข้ามตำแหน่ง)

## `__init__` — เตรียม 3 linear layers

```python
self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
```

จุดสังเกตแรก: **มี 3 linear layer ไม่ใช่ 2 อย่าง MLP ทั่วไป** (`Linear → activation → Linear`) นี่คือความต่างสำคัญของ SwiGLU เทียบกับ MLP แบบเดิม (GPT-2 style)

- `gate_proj`: `hidden_size → intermediate_size` (เช่น 1024 → 4096) — เส้นทาง "gate" ที่จะผ่าน activation
- `up_proj`: `hidden_size → intermediate_size` เหมือนกัน — เส้นทาง "value" ที่**ไม่ผ่าน** activation
- `down_proj`: `intermediate_size → hidden_size` — โปรเจกกลับมาขนาดเดิมเพื่อบวก residual ได้

`bias=False` ทั้งหมด — LLM สมัยใหม่ส่วนใหญ่ตัด bias ทิ้งเพื่อลด parameter และงานวิจัยพบว่าแทบไม่กระทบ performance

```python
self.act_fn = ACT2FN[config.hidden_act]
```
`ACT2FN` เป็น dict lookup ของ transformers เช่น `{"silu": nn.SiLU(), "gelu": nn.GELU(), ...}` — Qwen3 ปกติใช้ `"silu"` (Sigmoid Linear Unit หรือเรียก Swish) ตรงนี้ทำให้เปลี่ยน activation function ได้แค่แก้ค่า config โดยไม่ต้องแก้โค้ด

## `forward` — สูตร SwiGLU

```python
def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj
```

แตกเป็นขั้นตอน:
1. `self.gate_proj(x)` → โปรเจก x ไป intermediate_size
2. `self.act_fn(...)` → ผ่าน SiLU activation
3. `self.up_proj(x)` → โปรเจก x เดิม (คนละ weight กับ gate) ไป intermediate_size เหมือนกัน **ไม่ผ่าน activation**
4. คูณ element-wise กันสองเส้น (`gate ผ่าน act` × `up ไม่ผ่าน act`) — นี่คือ "gating mechanism" เส้น gate ทำหน้าที่เหมือนสวิตช์เปิด-ปิดว่า dimension ไหนควรผ่านไปเยอะ/น้อย
5. `self.down_proj(...)` → บีบกลับมาขนาด `hidden_size` เดิม

สูตรคณิตศาสตร์:
$$\text{SwiGLU}(x) = \big(\text{SiLU}(xW_{gate}) \odot xW_{up}\big)W_{down}$$

โดย SiLU คือ:
$$\text{SiLU}(x) = x \cdot \sigma(x)$$

## ทำไมต้องมี 2 เส้น (gate + up) แทนที่จะเป็น MLP ธรรมดา

MLP แบบเก่า (GPT-2):
```python
def forward(self, x):
    return self.down_proj(self.act_fn(self.up_proj(x)))  # แค่ 2 linear
```

ปัญหาคือ activation function อย่าง ReLU/GELU **ตัดข้อมูลทิ้งแบบตายตัว** (ReLU ตัดค่าติดลบทิ้งหมด) ส่วน gating mechanism ของ SwiGLU ให้โมเดล**เรียนรู้เองว่าจะเปิด-ปิด dimension ไหนมากน้อยแค่ไหน** แบบ dynamic ผ่าน gate เส้นแยก — งานวิจัย (Noam Shazeer, "GLU Variants Improve Transformer") พบว่าให้ performance ดีขึ้นกว่า MLP ธรรมดาอย่างสม่ำเสมอ โดยแลกด้วย parameter เพิ่มขึ้นจาก 3 matrix แทน 2

## Trade-off ที่ควรรู้เวลาจะปรับ config เอง
เพราะมี 3 matrix (`gate`, `up`, `down`) แทน 2 (`up`, `down`) แบบเดิม พอ fix parameter budget เท่ากัน คนมักลด `intermediate_size` ลงชดเชย (Llama/Qwen ใช้ intermediate_size ประมาณ 2.7-3.5 เท่าของ hidden_size แทนที่จะเป็น 4 เท่าแบบ GPT เดิม) ถ้าจะ custom `YaeConfig` เอง ตรงนี้เป็นจุดที่ต้อง tune ให้สมดุลระหว่าง capacity กับ parameter count

ตรงๆ เลยคือ **ไม่มีจุดไหนใน MLP ที่ "เข้าใจ" อะไรเป็นชิ้นเป็นอันเดี่ยวๆ** — นี่คือจุดที่คนเข้าใจผิดกันบ่อย เพราะเราชอบมองหา "จุดที่ปัญญาอยู่" แบบมีตำแหน่งชัดเจน แต่ในความเป็นจริงมันกระจายอยู่ใน **weight values** (ตัวเลขในเมทริกซ์) ไม่ใช่ในตัว operation

แต่ถ้าถามว่า MLP มี **บทบาทหน้าที่** อะไรต่างจาก Attention ในการ "เข้าใจ" — อันนี้ตอบได้ชัดกว่า

## MLP vs Attention: หน้าที่ต่างกันคนละมิติ

**Attention** (ที่เคยดู `Qwen3Attention` กันไปแล้ว) ทำหน้าที่ **"token คุยกันข้ามตำแหน่ง"** — เช่นคำว่า "มัน" ในประโยคจะ attend กลับไปหาว่า "มัน" หมายถึงคำไหนก่อนหน้า นี่คือกลไกที่จับ **ความสัมพันธ์ระหว่างคำ** (syntax, coreference, context)

**MLP** ที่ดูกันไปเมื่อกี้ ทำงาน **แยกทีละ token อิสระ** (สังเกตว่า `forward(self, x)` ไม่มีการข้าม sequence dimension เลย ทำกับแต่ละตำแหน่งแยกกันเป๊ะๆ) — นี่คือจุดที่งานวิจัยเรียกว่า **"key-value memory"**

## MLP เป็นเหมือนฐานข้อมูลความรู้แบบ fuzzy

มีงานวิจัยชื่อ *"Transformer Feed-Forward Layers Are Key-Value Memories"* (Geva et al.) ที่ชี้ให้เห็นว่า:

- `gate_proj`/`up_proj` (เส้นที่ยกมิติขึ้นไป `intermediate_size`) ทำหน้าที่คล้าย **"key"** — แต่ละ neuron ใน intermediate layer จะ activate แรงเมื่อ input ตรงกับ pattern เฉพาะบางอย่าง เช่น neuron หนึ่งอาจ activate แรงเวลาเจอ context เกี่ยวกับ "เมืองหลวงของประเทศ"
- `down_proj` ทำหน้าที่คล้าย **"value"** — พอ neuron นั้น activate ก็ inject ข้อมูลที่เกี่ยวข้องกลับเข้าไปใน residual stream เช่น inject ทิศทางที่โน้มไปทาง token "ปารีส" ถ้า context คือ "เมืองหลวงของฝรั่งเศส"

พูดง่ายๆ: **Attention บอกว่า "ต้องดูตรงไหน" ส่วน MLP บอกว่า "รู้อะไรเกี่ยวกับสิ่งนั้นบ้าง"** งานวิจัยหลายชิ้นพบว่า factual knowledge (เช่น "ปารีสเป็นเมืองหลวงฝรั่งเศส") ไปฝังอยู่ใน MLP weight เป็นส่วนใหญ่ ไม่ใช่ Attention

## ทำไมถึงไม่มี "จุดเดียว" ที่เข้าใจ

เพราะ `intermediate_size` มีนับพัน-หมื่น neuron ต่อ layer และมี layer ซ้อนกัน 32+ ชั้น "ความเข้าใจ" ไม่ได้อยู่ที่ neuron ตัวใดตัวหนึ่ง แต่อยู่ที่ **pattern ของการ activate ร่วมกันของ neuron หลายพันตัว across หลายสิบ layer** พร้อมกัน — เป็น distributed representation ไม่ใช่ localized คล้ายๆ กับที่สมองคนก็ไม่มี "เซลล์เดียวที่รู้ว่าแม่คุณหน้าตายังไง" แต่เป็น pattern กระจายในเครือข่ายเซลล์ประสาทเยอะๆ พร้อมกัน

เพราะงั้นถ้าอยากรู้ว่า YAE "รู้" อะไรจริงๆ วิธีเดียวที่ทำได้แม่นคือดูจาก **behavior หลัง train** ไม่ใช่ไปไล่อ่าน weight matrix ทีละตัวแล้วบอกว่า "อันนี้แปลว่าอะไร" — ต่อให้เป็นนักวิจัย interpretability มืออาชีพก็ยังทำได้แค่บางส่วนเท่านั้นในปัจจุบัน