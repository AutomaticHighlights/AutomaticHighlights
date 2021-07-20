# Course project for Deep Learning 2021 Spring:
# Automatic Football Highlights Editor 

## Introduction

In this course project, we develop a deep learning model to generate football highlights from the full game video automatically, using both video and audio information.

In specific, we design an automatic highlight editor model with two stages. The first stage is called scene classifier, used to detect essential scenes from a full game video. The second stage is called precise scene editor, which can precisely find the start point and endpoint for each scene detected by the scene classifier.

The code of training process can be found in `src/`. Besides, we pretrain two image feature extractors during developing our model. The code of pretraining can be found in `pretrain/`.

Technical details are described in [Final Report](FinalReport.pdf).

## Demos

We test our model on some recent football games. Here are the results. You can click the following links to watch the demos:

### Good cases

[FIFA World Cup Qatar 2022 Qualifiers: Guam 0-7 China](https://cloud.tsinghua.edu.cn/d/83dcfc3d2a1546818cb9/files/?p=%2F%5BWCQ2022%5DGuam_vs_China_2021-05-30.mp4) 2021.05.30

[FIFA World Cup Qatar 2022 Qualifiers: China 3-1 Syria](https://cloud.tsinghua.edu.cn/d/83dcfc3d2a1546818cb9/files/?p=%2F%5BWCQ2022%5DChina_vs_Syria_2021-06-16.mp4) 2021.06.16

[Copa América 2021: Brazil 4-0 Peru](https://cloud.tsinghua.edu.cn/d/83dcfc3d2a1546818cb9/files/?p=%2F%5BCA2021%5DBrazil_vs_Peru_2021-06-18.mp4) 2021.06.18

[Copa América 2021: Colombia 1-2 Peru](https://cloud.tsinghua.edu.cn/d/83dcfc3d2a1546818cb9/files/?p=%2F%5BCA2021%5DColombia_vs_Peru_2021-06-21.mp4) 2021.06.21

### Bad cases

[UEFA Champions League 2020/21 Final: Manchester City 0-1 Chelsea](https://cloud.tsinghua.edu.cn/d/83dcfc3d2a1546818cb9/files/?p=%2F%5BUCL20-21%5DManchesterCity_vs_Chelsea_2021_05_30.mp4) 2021.05.30

[FIFA World Cup Qatar 2022 Qualifiers: China 2-0 Philippines](https://cloud.tsinghua.edu.cn/d/83dcfc3d2a1546818cb9/files/?p=%2F%5BWCQ2022%5DChina_vs_Philippines_2021-06-08.mp4) 2021.06.08

[FIFA World Cup Qatar 2022 Qualifiers: China 5-0 Maldives](https://cloud.tsinghua.edu.cn/d/83dcfc3d2a1546818cb9/files/?p=%2F%5BWCQ2022%5DChina_vs_Maldives_2021-06-12.mp4) 2021.06.12

[Copa América 2021: Argentina 1-1 Chile](https://cloud.tsinghua.edu.cn/d/83dcfc3d2a1546818cb9/files/?p=%2F%5BCA2021%5DArgentina_vs_Chile_2021-06-15.mp4) 2021.06.15

[Copa América 2021: Paraguay 3-1 Bolivia](https://cloud.tsinghua.edu.cn/d/83dcfc3d2a1546818cb9/files/?p=%2F%5BCA2021%5DParaguay_vs_Bolivia_2021-06-15.mp4) 2021.06.15


PS: It's a pity that the videos of UEFA Euro 2020 (hold in 2021) are not released in CCTV.com, hence we didn't test our model on it.
