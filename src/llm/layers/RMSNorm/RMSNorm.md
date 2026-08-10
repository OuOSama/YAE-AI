มาไล่ทีละส่วนเลย class นี้คือ **RMSNorm** (Root Mean Square Normalization) — ตัว normalization layer ที่ Qwen3 (และ LLM สมัยใหม่แทบทุกตัว) ใช้แทน LayerNorm แบบเดิม

## `__init__` — ตั้งค่าเริ่มต้น

```python
def __init__(self, hidden_size, eps: float = 1e-6) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
```

- `self.weight = nn.Parameter(torch.ones(hidden_size))` — สร้าง learnable parameter ขนาดเท่า `hidden_size` เริ่มต้นเป็น 1 ทั้งหมด (เวก tor scale ที่โมเดลจะปรับระหว่างเทรน) `nn.Parameter` ทำให้ PyTorch รู้ว่านี่คือ weight ที่ต้องอัพเดตตอน backprop
- `self.variance_epsilon = eps` — ค่าเล็กๆ (default `1e-6`) กันหาร 0 ตอนคำนวณ `rsqrt`

**สังเกตว่าไม่มี `bias` เลย** — ต่างจาก LayerNorm ทั่วไปที่มีทั้ง scale (`gamma`) และ shift (`beta`) RMSNorm ตัดส่วน mean-shift ทิ้งไปเลย เหลือแค่ scale อย่างเดียว นี่คือเหตุผลที่มันเร็วกว่า LayerNorm

## `forward` — คำนวณจริง

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)
```

ไล่ทีละบรรทัด:

1. **`input_dtype = hidden_states.dtype`** — จำ dtype เดิมไว้ก่อน (ปกติจะเป็น `bf16` หรือ `fp16` ตอนเทรน)

2. **`hidden_states = hidden_states.to(torch.float32)`** — บังคับ cast ขึ้นเป็น `float32` ชั่วคราว เพราะการคำนวณ variance/rsqrt ถ้าทำใน `bf16` จะเสีย precision ง่ายมาก (bf16 มี mantissa สั้น) ทำให้ training ไม่เสถียร นี่คือเหตุผลว่าทำไม RMSNorm ถึงมักเป็นจุดที่ "เปราะ" ต่อ numerical instability ถ้าลืม cast ตรงนี้

3. **`variance = hidden_states.pow(2).mean(-1, keepdim=True)`** — หัวใจของ RMS: ยกกำลังสองทุกค่าใน hidden dimension สุดท้าย แล้วหาค่าเฉลี่ย (นี่คือ "mean square" — ไม่ลบ mean ออกก่อนแบบ variance ปกติ ต่างจาก LayerNorm ตรงนี้แหละ) `keepdim=True` เพื่อให้ broadcast กลับไปคูณกับ tensor เดิมได้พอดี

4. **`hidden_states * torch.rsqrt(variance + self.variance_epsilon)`** — นี่คือ normalize: หาร hidden_states ด้วย root-mean-square (บวก epsilon กันหาร 0) `rsqrt(x)` เทียบเท่า `1/sqrt(x)` แต่เร็วกว่าเพราะ GPU มี hardware instruction เฉพาะสำหรับมัน

5. **`self.weight * hidden_states.to(input_dtype)`** — cast กลับไป dtype เดิม (`bf16`) แล้วคูณด้วย learnable scale — ได้ output สุดท้าย

## `extra_repr` — แค่ debug helper

```python
def extra_repr(self):
    return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
```
ไม่เกี่ยวกับการคำนวณเลย แค่ทำให้ตอน `print(model)` โชว์ shape/eps ของ layer นี้อ่านง่ายขึ้น เช่น `YaeRMSNorm((1024,), eps=1e-06)`

## สูตรคณิตศาสตร์รวม
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum x_i^2 + \epsilon}} \cdot \gamma$$

เทียบกับ LayerNorm ที่ต้องลบ mean ก่อน (`(x - mean) / std`) RMSNorm ตัดขั้นตอนลบ mean ทิ้งไป — งานวิจัยพบว่า mean-centering ไม่ค่อยจำเป็นสำหรับ LLM เท่าไหร่ ตัด step นี้ทิ้งได้ compute เร็วขึ้นโดย performance แทบไม่ต่างเลย นี่คือเหตุผลหลักที่ LLM ยุคใหม่แทบทุกตัว (Llama, Qwen, Mistral) เปลี่ยนมาใช้ RMSNorm แทน LayerNorm กันหมด