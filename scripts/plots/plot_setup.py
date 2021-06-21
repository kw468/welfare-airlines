import matplotlib.pyplot as plt
import seaborn as sns

FONT = "Liberation Serif"
FONT_SIZE = 20
CSFONT = {"fontname": FONT, "fontsize": FONT_SIZE}
PALETTE = ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
LINE_WIDTH = 3
sns.set(style = "white", color_codes = False)
plt.rcParams["font.family"] = FONT
FIG_SIZE = (1.5 * 6.4, 1.1 * 4.8)
FIG_DPI = 600
FIG_FORMAT = "PDF"
