# colab_notebooks

## Summary
- analysis
- animation
- audio
- generative
- meeting
- motion
- object_detection
- searchsystem
- semantic_segmentation
- vision3d

## Contents
* Analysis
  * [ptlflow](analysis/ptlflow_inference.ipynb)
    - opticalflowを代表とした画像内の動きを推論できるAIモデルのNotebook
  * [face detection](analysis/face_detection.ipynb)
* Animation
  * [anime_rad_nerf](animation/anime_rad_nerf.ipynb)
    - 顔の映像と音声を合成する.アニメ映像でできるように改変
  * [RAD_NeRF](animation/RAD_NeRF.ipynb)
    - 顔の映像と音声を合成する。3Dの推論を入れてるので精度が高い
  * [Thin Plate Spline Motion](animation/Thin_Plate_Spline_Motion_Model_original.ipynb)
    - 顔の映像を元映像の動きと同期させて話せるようになっている。
* Audio
  * [audiolm](audio/audiolm.ipynb)
    - 音楽を生成できるやつ
  * [DeforumStableDiffusion](audio/Deforum_Stable_Diffusion_Mubert_original.ipynb)
    - 音楽生成が可能（BGMより）、動画との組み合わせも可能
  * [Denoise](audio/Denoiser_fb_examples.ipynb)
    - 雑音除去の性能が高いやつ
  * [riffusion](audio/riffusion.ipynb)
    - 音楽生成が可能（BGMより）、fine-tuningコードあり
  * [valle](audio/vall_e.ipynb)
    - 誰かの声真似ができる
* Generative
  * [img2prompt](generative/img2prompt.ipynb)
    - 画像からその画像の意味を抽出可能
  * [stable diffusion v2 finetune](generative/stable_diffusion_v2_finetuning.ipynb)
    - stable diffusion v2のdreambooth finetuningのやつ
  * [stable diffusion image inpaint](generative/StableDiffusion_image_inpainting.ipynb)
    - stable diffusionのinpaintingのdreambooth finetuningのやつ
  * [stable diffusion v1.5 inpaint](generative/stable_diffusion_inpaint_dreambooth_v1_5.ipynb)
  * [stable diffusion v2 inpaint](generative/stable_diffusion_inpainting_v2.ipynb)
* Meeting
  * [meeting recognition](meeting/meeting.ipynb)
    - whisper + pyannoteで話者識別、書き起こし、音声類似度判定で、誰が喋ったかまで可能
* Motion
  * [alphapose3d](motion/alphapose_master_3d.ipynb)
    - 3d 姿勢推定ライブラリの実行
  * [motiondiffusion](motion/MotionDiffuse_original.ipynb)
    - textからアクションを生成可能(ToDo: unityで使えるように連携[記事](https://note.com/npaka/n/nc76278c4a646))
  * [motion diffusion unity](motion/motion_diffusion_unity.ipynb)
    - text2unityのためのipynb
* ObjectDetection
  * [visionTransformer](object_detection/vision_transformer_finetuning.ipynb)
    - vision transformerで物体検出、自分用にfine-tuningする手順
  * [cutler](object_detection/cutler.ipynb)
    - 教師なしでマスク検出が可能、detectron2には大きく依存だが、ここのdetectionをfine-tuningすれば自分用にカスタマイズ可能かも
* SearchSystem
  * [finetuner](searchsystem/genshin_finetuner_search_system.ipynb)
* SemeanticSegmentation
  * [unetdeeplab](semantic_segmentation/unet_deeplabv3.ipynb)
    - unet + deeplabv3でsemantic segmentationを自分用でfinetuningする手順
* Vision3d
  * [ECON](vision3d/ECON_origin.ipynb)
  * [get3d](vision3d/get3d.ipynb)
  * [ICON](vision3d/ICON_train.ipynb)
  * [latentNerf](vision3d/latent_nerf.ipynb)
  * [NerfStudio](vision3d/nerfstudio.ipynb)
  * [visionNerf](vision3d/vision_nerf.ipynb)
* App
  * [gradioapp samples](app/gradioapp.ipynb)
* Tools
  * [instructpix2pix dataset creation](tools/instructpix2pix2_dataset.ipynb)
* Video
  * [Tune a video](video/tune_a_video.ipynb)


## Tips
[Collaboration with github and colaboratory](https://hirotaka-hachiya.hatenablog.com/entry/2019/06/10/000051)

## TODo:
* mask interaction system
    - [GLIGEM](https://github.com/gligen/GLIGEN) [Demo](https://huggingface.co/spaces/gligen/demo)
    - [LaMa](https://huggingface.co/spaces/akhaliq/lama)



