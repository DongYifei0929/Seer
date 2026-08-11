
# 3. Basic packaging tools
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==57.5.0"

# 4. Install PyTorch per Seer official docs
python -m pip install \
torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
--index-url https://download.pytorch.org/whl/cu121

# 5. Install Seer requirements
python -m pip install -r requirements.txt

# 6. Extra packages needed by the RoboLab bridge script
python -m pip install \
wandb==0.16.6 \
scipy==1.10.1 \
pandas==2.0.3 \
pyarrow==14.0.1 \
opencv-python==4.11.0.86

# 7. Check imports
python - <<'PY'
import torch, torchvision, clip, timm, wandb, pandas, cv2, scipy
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("Seer env OK")
PY