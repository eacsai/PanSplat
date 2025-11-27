python -m src.main +experiment=pansplat-256 ++model.weights_path=logs_good/two_view_mp3d/checkpoints/last.ckpt mode=test test.compute_scores=true

python -m src.main +experiment=pansplat-160 ++model.weights_path=logs_good/two_view_mp3d/checkpoints/last.ckpt mode=test test.compute_scores=true

# Train
python -m src.main +experiment=pansplat-160 mode=train

## MP3D
python -m src.main +experiment=pansplat-256 ++model.weights_path=logs_good/two-view-mp3d-256/checkpoints/last.ckpt mode=test test.compute_scores=true

python -m src.main +experiment=pansplat-512 ++model.weights_path=logs_good/two-view-mp3d-256/checkpoints/last.ckpt mode=test test.compute_scores=true

## 360Loc
python -m src.main +experiment=pansplat-160-360loc ++model.weights_path=logs_good/ls933m5x/checkpoints/last.ckpt mode=test test.compute_scores=true

python -m src.main +experiment=pansplat-160-360loc ++model.weights_path=logs_good/two_img_360loc/checkpoints/last.ckpt mode=test test.compute_scores=true

python -m src.main +experiment=pansplat-256-360loc ++model.weights_path=logs_good/two_img_360loc/checkpoints/last.ckpt mode=test test.compute_scores=true

python -m src.main +experiment=pansplat-512-360loc ++model.weights_path=logs_good/ls933m5x/checkpoints/last.ckpt mode=test test.compute_scores=true

## VIGOR
python -m src.main +experiment=pansplat-160-VIGOR ++model.weights_path=logs_good/vigor/checkpoints/last.ckpt mode=test test.compute_scores=true