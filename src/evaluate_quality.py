from evaluate import compute_fid

# -------- CONFIG --------
real_dir = "../dermDatabaseOfficial/release_v0/images"
synthetic_dir = "../synthetic_dataset"
# ------------------------

print("\n🔍 Evaluating FID and SSIM between real and synthetic images...\n")

# FID
fid_score = compute_fid(real_dir, synthetic_dir)
print(f"📊 FID Score: {fid_score:.4f} ")

