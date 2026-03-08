---
layout: default
title: "Training-based versus training-free differential privacy for data synthesis"
---

<div class="header">
  <h1 style="counter-increment: none;">Training-based versus training-free<br>differential privacy for data synthesis</h1>
  <div class="authors">
    <a href="https://github.com/mekapur">Mehak Kapur</a>,
    <a href="https://github.com/hanajuliatj">Hana Tjendrawasi</a>,
    <a href="https://github.com/jktrns">Jason Tran</a>,
    <a href="https://github.com/21phuctran">Phuc Tran</a>
  </div>
  <div class="emails">{mekapur,htjendrawasi,jat037,pct001}@ucsd.edu</div>
  <div class="affiliation">
    Halıcıoğlu Data Science Institute, UC San Diego<br>
    Mentor: Yu-Xiang Wang
  </div>
  <div class="links">
    <a href="https://github.com/mekapur/DSC180B-Q2">Code</a>
    <a href="#">Report (PDF)</a>
    <a href="#">Poster</a>
  </div>
</div>

<div class="abstract" markdown="1">

Differentially private synthetic data generation promises to resolve the tension between data utility and individual privacy, enabling the release of datasets that preserve the statistical properties analysts need while bounding what any adversary can learn about a single record. Two paradigms have emerged: training-based methods, which inject calibrated noise during model optimization, and training-free methods, which leverage foundation models through black-box API access. We investigate both approaches on Intel's Driver and Client Applications (DCA) telemetry corpus, evaluating against a benchmark of 21 analytical SQL queries representative of production business intelligence workloads.

</div>

<section id="introduction" markdown="1">

## Introduction

### Why does this matter?

### The privacy-utility tension

### Two paradigms for synthetic data

</section>

<section id="data" markdown="1">

## Data

### Intel DCA telemetry corpus

### The 21-query SQL benchmark

### Formal benchmark definition

</section>

<section id="methods" markdown="1">

## Methods

### Training-based: DP-SGD with variational autoencoders

### Training-free: Private Evolution via foundation model APIs

### MST baseline

### Per-table synthesis

### Privacy budget comparison

</section>

<section id="results" markdown="1">

## Results

### Cross-method comparison

### Performance by query type

### Calibration analysis

### Sensitivity to evaluation thresholds

</section>

<section id="discussion" markdown="1">

## Discussion

### What worked

### What didn't

### Why PE underperforms on tabular data

### Future directions

</section>

## References
