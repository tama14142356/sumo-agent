[日本語での説明](#gym環境用の学習コード)  
[explanation in English](#learning-code-for-gym-environment)
# gym環境用の学習コード

## 必須システム要件
OS : Linux, Windows7以上, macOS  
(ただし、動作確認したのはUbuntu18.04LTSのみ)  
ソフトウェア : SUMO, python3  
python環境 : anaconda を使ったpython3.7以上  
GPU: cuda10.2 (なくてもいい)

## SUMO・gym環境のインストール方法
SUMO・gym環境のインストール方法は
[SUMO・gym環境のインストール方法](https://git.esslab.jp/tomo/gym-sumo/-/blob/master/README.md)を見てください。

## 学習環境のインストール
1. 前セクションでSUMO用のgym環境を整えておく。
1. 学習環境のソースコードを取得する。  
    もし、ssh公開鍵をgitlabに登録しているなら、

      ```
      $ cd ~
      $ git clone --recursive git@git.esslab.jp:tomo/sumo-agent.git
      ```

    登録していないなら、

      ```
      $ cd ~
      $ git clone --recursive https://git.esslab.jp/tomo/sumo-agent.git
      ```
1. 実行するのに必要なライブラリ、パッケージをインストール  
    CUDAが使えるGPUが搭載されているなら、以下のコマンドを実行  
    (anaconda環境であることを前提とします。)

      ```
      $ cd ~/sumo-agent
      $ conda create -n sumogpu python=3.8 -y && conda activate sumogpu
      $ bash install_gpu.sh
      ```

    cpuしか使えないなら、以下のコマンドを実行

      ```
      $ cd ~/sumo-agent
      $ bash install.sh
      ```

## 学習コード実行例
   ```
   $ cd ~/sumo-agent
   $ bash library_sample/pfrl_sample/gym_example_original/test_gym/test_dqn_cpu.sh
   ```

# learning code for gym environment

## prerequirement
OS : Linux, Windows7+  
(confirmed operation with Ubuntu18.04LTS)  
software : SUMO, python3  
python environment : anaconda python 3.7+  
GPU:CUDA10.2(don't need)

### install SUMO/gym environment
the way to install SUMO/gym environment is 
[install SUMO/gym environment](https://git.esslab.jp/tomo/gym-sumo/-/blob/master/README.md)

## install agent environment
1. install SUMO/gym environment by following the previous section
1. install SUMO by following the previous section
1. get source code  
    if ssh public key is registered on gitlba, run the following command:
    
      ```
      $ cd ~
      $ git clone --recursive git@git.esslab.jp:tomo/sumo-agent.git
      ```
    
    if ssh public key is not registered, run the following command:
    
      ```
      $ cd ~
      $ git clone --recursive https://git.esslab.jp/tomo/sumo-agent.git
      ```
1. install requirement packages:  
   if your system has gpu using by cuda driver, run the following command  
   (python environment presupposes anaconda)

    ```
    $ cd ~/sumo-agent
    $ conda create -n sumogpu python=3.8 -y && conda activate sumogpu
    $ bash install_gpu.sh
    ```

   if your system has only cpu, run the following command

    ```
    $ cd ~/sumo-agent
    $ bash install.sh
    ```

## Usage
   ```
   $ cd ~/sumo-agent
   $ bash library_sample/pfrl_sample/gym_example_original/test_gym/test_dqn_cpu.sh
   ```
