import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 基于脚本位置构造文件路径（假设数据在项目的 src/ 目录下）
BASE_DIR = Path(__file__).resolve().parent
file1 = BASE_DIR / 'src' / 'Output_Spectrum.txt'
file2 = BASE_DIR / 'src' / 'Proton_spectrum.txt'

def load_two_files(f1: Path, f2: Path):
	missing = [str(p) for p in (f1, f2) if not p.exists()]
	if missing:
		print('文件未找到：', ', '.join(missing), file=sys.stderr)
		print('请确认这些文件位于脚本目录的 src/ 子目录，或修改路径。', file=sys.stderr)
		raise FileNotFoundError(missing[0])
	e1, fl1 = np.loadtxt(f1, comments='#').T
	e2, fl2 = np.loadtxt(f2, comments='#').T
	return (e1, fl1), (e2, fl2)


try:
	(energy1, flux1), (energy2, flux2) = load_two_files(file1, file2)
except FileNotFoundError:
	raise

# 绘图
plt.figure(figsize=(10, 7))
plt.loglog(energy1, flux1, 'b-', linewidth=2, label='Output Spectrum')
plt.loglog(energy2, flux2, 'r-', linewidth=2, label='Proton Spectrum')

plt.xlabel('Energy (GeV)', fontsize=14)
plt.ylabel('Flux', fontsize=14)
plt.title('Comparison', fontsize=16)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
plt.show()