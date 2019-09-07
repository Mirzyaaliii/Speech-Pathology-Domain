# Scripts

Scripts for feature extraction from audio file, create batches and
audio synthesis.

##### Flow for Voice Conversion

  1. aho_features.sh
  2. concatenation.m
  3. create_batches.m
  4. Feature_extraction_testing.m
  5. testmat2mcc_conversion.m
  6. aho_synth.sh

##### 1. aho_features.sh
 - Extract features from audio file (.wav file).
 - If audio file is not in 16K hz SoX player convert it in 16K hz.
(SoX reads and writes audio files in most popular formats and can
optionally apply effects to them.)
 - Ahocoder extract feature from audio file (f0, mcc, fv). (Ahocoder
parameterizes speech waveforms into three different streams: log-f0,
cepstral representation of the spectral envelope, and maximum voiced
frequency.)

##### 2. concatenation.m
 - Concatenate cepstral features of source and target speaker.
 - First it align features of source and target speaker using dtw_E.m
(Dynamic Time Warping algorithm) and store it to X and Y respectively.
 - Z.mat contains aligned features. (size of Z.mat is 80Xa, a
depends on no. of files.)

#####  3. create_batches.m
  - Create batches of 1000X40 from Z.mat.

#####  4. Feature_extraction_testing.m
   - Create batches for testing using cepstral features. (size of
testing file is bX40, b depends on length of file.)

#####  5. testmat2mcc_conversion.m
  - Convert mat files back to mcc and save it with the same name of
original mcc features.

#####  6. aho_synth.sh
  - Synthesis audio file from converted mcc and f0 using ahodecoder.
