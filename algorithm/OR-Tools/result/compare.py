import pandas as pd
import matplotlib.pyplot as plt

# 1️ Đọc file CSV tổng hợp kết quả
df = pd.read_csv("algorithm/OR-Tools/result/ortools_summary.csv")

# 2️ Trích xuất nhóm dữ liệu (C, R, RC)
df["Category"] = df["Dataset"].str.extract(r"([A-Z]+[0-9])")[0].str.replace(r"\d+", "", regex=True)

# 3️ Tính trung bình theo nhóm
summary = (
    df.groupby("Category")
    .agg(
        Avg_Distance=("Total_Distance", "mean"),
        Avg_Vehicles=("Num_Vehicles", "mean")
    )
    .round(2)
    .reset_index()
)

# 4️ Vẽ biểu đồ kết hợp bar (quãng đường) + line (số xe)
fig, ax1 = plt.subplots(figsize=(8, 5))

# Cột: Quãng đường trung bình
color1 = "tab:blue"
ax1.bar(summary["Category"], summary["Avg_Distance"], color=color1, alpha=0.7, label="Quãng đường TB (km)")
ax1.set_xlabel("Nhóm dữ liệu Solomon", fontsize=12, fontweight="bold")
ax1.set_ylabel("Quãng đường trung bình (km)", color=color1, fontsize=12, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# Đường: Số xe trung bình
ax2 = ax1.twinx()
color2 = "tab:orange"
ax2.plot(summary["Category"], summary["Avg_Vehicles"], color=color2, marker="o", linewidth=2.5, label="Số xe TB")
ax2.set_ylabel("Số xe trung bình", color=color2, fontsize=12, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=color2)

# 5️ Tiêu đề và định dạng
plt.title("So sánh quãng đường và số xe trung bình giữa các nhóm dữ liệu Solomon",
          fontsize=14, fontweight="bold", pad=10)
fig.tight_layout()

# 6️ Lưu hình (tuỳ chọn)
plt.savefig("ortools_group_result.png", dpi=300, bbox_inches="tight")
plt.show()
