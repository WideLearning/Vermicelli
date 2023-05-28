import re
from random import randint


def random_name():
    return "t_" + hex(randint(0, 16**8 - 1))[2:]


rules = [
    (r"\s*//.*", r""),
    (r"vector (.*)", r"\1 ~ N(v(0, 0), v(1e3, 1e3))"),
    (r"real (.*)", r"\1 ~ N(0.0, 1e3)"),
    (r"v\((.*?), (.*?)\)", r"tensor([\1, \2], dtype=torch.float)"),
    (r"d\((.*?), (.*?)\)", r"torch.hypot(\1[0] - \2[0], \1[1] - \2[1])"),
    (r"(.*) ~ (.*) = (.*)", r"\1 = pyro.sample('\1', \2, obs=\3)"),
    (r"(.*) ~ (.*)", r"\1 = pyro.sample('\1', \2)"),
    (r"N\((.*),(.*)\)", r"dist.Normal(\1,\2)"),
]

code = """
// anchors
OX_len ~ N(90.0, 1.0)
X := -OX_len * v(0, 1)
M ~ N(v(10, 55), v(2, 3))

// nose
n1 ~ N(v(0, 20), v(5, 5))
n2 ~ N(v(40, 0), v(4, 4))
n2s := MIRROR_X @ n2
n3 ~ N(v(30, -20), v(4, 4))
n3s := MIRROR_X @ n3

// lips
Mmb ~ N(0.2 * M, v(2, 2))
Mma ~ N(0.8 * PERP_LEFT @ M, v(5, 5))
ma := M + Mma
mas := M - Mma
mb := M + Mmb
mbs := M - Mmb

// face oval
Xc3 ~ N(v(155, 0), v(10, 20))
Xc2 ~ N(v(122, 120), v(10, 10))
Xc1 ~ N(v(35, 220), v(15, 10))
c3r := X + Xc3
c2rd := X + Xc2
c1rd := X + Xc1
c1ld := X + MIRROR_X @ Xc1
c2ld := X + MIRROR_X @ Xc2
c3l := X - Xc3
c2lu := X - Xc2
c1lu := X - Xc1
c1ru := X + MIRROR_Y @ Xc1
c2ru := X + MIRROR_Y @ Xc2

// eye centers
Xe ~ N(v(80, 0), v(1, 1))
er := X + Xe
el := X + MIRROR_X @ Xe

// eyebrows
eb1 ~ N(v(-40, -40), v(5, 5))
eb2 ~ N(v(0, -50), v(5, 5))
eb3 ~ N(v(40, -40), v(5, 5))
b1r := er + eb1
b2r := er + eb2
b3r := er + eb3
b1l := el + MIRROR_X @ eb1
b2l := el + MIRROR_X @ eb2
b3l := el + MIRROR_X @ eb3

// eyes
ee1 ~ N(v(-30, 10), v(2, 2))
ee2 ~ N(v(-20, -20), v(4, 4))
ee3 ~ N(v(26, -8), v(4, 4))
ee4 ~ N(v(30, 7), v(5, 3))
e1r := er + ee1
e2r := er + ee2
e3r := er + ee3
e4r := er + ee4
e1l := el + MIRROR_X @ ee1
e2l := el + MIRROR_X @ ee2
e3l := el + MIRROR_X @ ee3
e4l := el + MIRROR_X @ ee4

// hair
"""

hairs = 2
for i in range(hairs):
    code += f"h{i}a ~ N(v(0, -290), v(10, 40))\n"
    code += f"h{i}b ~ N(c2ru, v(20, 20))\n"
    code += f"h{i}be ~ N(0, v(30, 30))\n"
    code += f"h{i}bs := h{i}be + MIRROR_X @ h{i}b\n"
    code += f"h{i}c ~ N(1.1 * c3r, v(20, 20))\n"
    code += f"h{i}ce ~ N(0, v(30, 30))\n"
    code += f"h{i}cs := h{i}ce + MIRROR_X @ h{i}c\n"
    code += f"h{i}d ~ N(1.5 * c2rd, v(50, 50))\n"
    code += f"h{i}de ~ N(0, v(60, 60))\n"
    code += f"h{i}ds := h{i}de + MIRROR_X @ h{i}d\n"

render = """"
plot("n3", "n2", "n1", "n2s", "n3s")
plot("mas", "mbs", "ma", "mb", "mas", "ma")
plot("b1r", "b2r", "b3r", "b1r")
plot("b1l", "b2l", "b3l", "b1l")
plot("c3r", "c2rd", "c1rd", "c1ld", "c2ld", "c3l")
plot("e1r", "e2r", "e3r", "e4r", "e1r")
plot("e1l", "e2l", "e3l", "e4l", "e1l")
"""

for i in range(hairs):
    render += f'plot("h{i}a", "h{i}b", "h{i}c", "h{i}d")\n'
    render += f'plot("h{i}a", "h{i}bs", "h{i}cs", "h{i}ds")\n'


def apply_subs(lines, rules):
    subs = 0
    for x, y in rules:
        for i in range(len(lines)):
            lines[i], n = re.subn(x, y, lines[i])
            subs += n
        lines = "\n".join(lines).split("\n")
        if subs:
            return lines, subs
    return lines, subs


code = code.split("\n")[1:]
render = render.split("\n")[1:]

while True:
    new_code = []
    for line in code:
        line = re.sub(r"#", random_name(), line)
        parts = line.split(" := ")
        if len(parts) == 2:
            rules.append((r"\b" + parts[0] + r"\b", "(" + parts[1] + ")"))
        else:
            new_code.append(line)
    if code == new_code:
        break
    else:
        code = new_code

while True:
    code, subs = apply_subs(code, rules)
    if not subs:
        break
while True:
    render, subs = apply_subs(render, rules)
    if not subs:
        break

print(*code, sep="\n")
code = "\n    ".join(code)
render = "\n        ".join(render)
with open("prob.py", "r") as f:
    template = f.read()
final_code = re.sub("pass  # model", code, template)
final_code = re.sub("pass  # render", render, final_code)
with open("a.py", "w") as f:
    f.write(final_code)
