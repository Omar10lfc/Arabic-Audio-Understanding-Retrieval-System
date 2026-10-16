with open("scratch/build_clean_n2_n3.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace any \n string literal in n2_c1 and n3_c1 with chr(10)
old_gpu = 'out_lines.append("GPU Info:\\n" + str(gpu_info).strip())'
new_gpu = 'out_lines.append("GPU Info:" + chr(10) + str(gpu_info).strip())'

old_write = 'f.write("\\n".join(out_lines) + "\\n")'
new_write = 'f.write(chr(10).join(out_lines) + chr(10))'

text = text.replace(old_gpu, new_gpu)
text = text.replace(old_write, new_write)

with open("scratch/build_clean_n2_n3.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated scratch/build_clean_n2_n3.py")
