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
- nlp
- japanesenlp
- app
- tools
- utilities
- pix2pix
- ocr
- inpainting
- instructionpix

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
  * [MakeItTalk](animation/makeittalk.ipynb)
    - 顔の映像と口元、目の当たりの動きをリンクさせて動かせるようなモデルになっている。Realistic visionも用いて生成した絵を動かすまでのパイプラインも紹介
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
  * [audio extraction](audio/audio_extraction.ipynb)
    - whisperとpyannoteを使って音声合成のためのデータセット作成を簡易化したノートブック
* Generative
  * [img2prompt](generative/img2prompt.ipynb)
    - 画像からその画像の意味を抽出可能
  * [stable diffusion v2 finetune](generative/stable_diffusion_v2_finetuning.ipynb)
    - stable diffusion v2のdreambooth finetuningのやつ
  * [stable diffusion image inpaint](generative/StableDiffusion_image_inpainting.ipynb)
    - stable diffusionのinpaintingのdreambooth finetuningのやつ
  * [stable diffusion v1.5 inpaint](generative/stable_diffusion_inpaint_dreambooth_v1_5.ipynb)
  * [stable diffusion v2 inpaint](generative/stable_diffusion_inpainting_v2.ipynb)
  * [stable diffusion for webui](generative/sd_webui.ipynb)
  * [openjourney](generative/openjourney.ipynb)
    - openjorney, nijijourneyの使い方、controlnetと組み合わせようとしたがgoogle colab freeのため失敗
    - [huggingface site](https://huggingface.co/prompthero/openjourney)
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
  * [mmsegtutorial PSPNet](semantic_segmentation/mmseg_tutorial.ipynb)
    - PSPNetで車載カメラデータセットをcityscapesのpretrainedでfinetuningする手順
  * [mmsegtutorial Deeplabv3](semantic_segmentation/mmseg_tutorial_deeplabv3.ipynb)
    - Deeplabv3で車載カメラデータセット、convert datasetの実験もつけてる
  * [unetseg](semantic_segmentation/UnetSeg.ipynb)
    - Unetを使ってsimpleなモデルで学習をおこなったケース
* Vision3d
  * [ECON](vision3d/ECON_origin.ipynb)
  * [get3d](vision3d/get3d.ipynb)
  * [ICON](vision3d/ICON_train.ipynb)
  * [latentNerf](vision3d/latent_nerf.ipynb)
  * [NerfStudio](vision3d/nerfstudio.ipynb)
  * [visionNerf](vision3d/vision_nerf.ipynb)
  * [text2mesh](vision3d/text2mesh.ipynb)
    - neural renderingを用いたtext2meshのデモ、ベースのmeshモデルによって精度は大きく変化するみたい。 -> remeshが走らない（なぜかメモリが足らない、、）
  * [live3d-v2](vision3d/live3d_v2.ipynb)
    - Neural renderingを用いて3Dモデルのモーションやモデル生成が可能、MDMと組み合わせて面白いことができそう。
  * [rgbd23d](vision3d/rgbd23d.ipynb)
    - midasで深度推定から3D point cloudを生成（全くうまくいかない）、そのほかにmmdetection3dのためのpoint cloud converterも実装されてる。

* Pix2Pix
  * [ControlNet](pix2pix/controlnet.ipynb)
  * [Pix2PixZero HuggingFace](pix2pix/huggingface_pix2pixzero.ipynb)
* App
  * [gradioapp samples](app/gradioapp.ipynb)
* Tools
  * [instructpix2pix dataset creation](tools/instructpix2pix2_dataset.ipynb)
* utilities
  * [mask2bbox](utilities/mask2bbox.ipynb)
    - [ref](https://dev.classmethod.jp/articles/make-bounding-box-from-mask-datas/)
  * [pytorch lightning](utilities/efficientnet_pytorch_lightning.ipynb)
    - efficient netを使った分類と値推定のNNの構築をpytorch lightningを使用して行ったデモ
* Video
  * [Tune a video](video/tune_a_video.ipynb)
* NLP
  * [GPT2 Finetuning](nlp/gpt2_finetuning_eng.ipynb)
* JapaneseNLP
  * [GPT2 Finetuning for Japanese](japanesenlp/huggingfacenlp.ipynb)
* Inpainting
  * [deepfillv2 demo](inpainting/deepfillv2.ipynb)
  * [latent-diffusion inpainting](inpainting/latent_diffusion_inpaint.ipynb)
* OCR
  * [OCR finetuning](ocr/deep_text_recognition_benchmark_original.ipynb)
* Pix2Pix for stable diffusion
  * [Controlnet](pix2pix/controlnet.ipynb)
* InstructionPix
  * [attend and excite](instructionpix/attendandexcite_huggingface.ipynb)
* LLM
  * [peft with huggingface](llm/peft.ipynb)
* chatgpt
  * [chatgpt with chatwaifu](chatgpt/chatwaifu.ipynb)
    - this is working in progress. maybe I cannot run on colab...
* Text2Speech
  * [vits finetuning](text2speech/vits_finetuning_original.ipynb)
* objectdetection3d
  * [mmdetection3d](objectdetection3d/mmdetection3d.ipynb)
    - mmdetection3dを用いた3d object detection、主にRGB+Point cloud or Point cloudの推定を行なっている。



## Tips
[Collaboration with github and colaboratory](https://hirotaka-hachiya.hatenablog.com/entry/2019/06/10/000051)

## TODo:
* mask interaction system
    - [GLIGEM](https://github.com/gligen/GLIGEN) [Demo](https://huggingface.co/spaces/gligen/demo)
    - [LaMa](https://huggingface.co/spaces/akhaliq/lama)

* 3D avatar creation
    - [Rodin diffusion](https://3d-avatar-diffusion.microsoft.com/)

* Stable diffusion paper
    - [awesome diffusion papers](https://github.com/heejkoo/Awesome-Diffusion-Models)
* Finetuning with blip
    - [blip huggingface](https://huggingface.co/docs/transformers/model_doc/blip)
    - [blip with dreambooth](https://github.com/KaliYuga-ai/blip-lora-dreambooth-finetuning/blob/main/KaliYuga_BLIP%2BLoRA%2BDreambooth_FineTuning.ipynb)
    - [Lora finetuning](https://huggingface.co/blog/lora)
    





