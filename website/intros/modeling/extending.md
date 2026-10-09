---
title: Extending Modeling
description: Add an architecture, a decision head, a training method, a reward or a whole recipe.
---

# Extending Modeling

Each family is a catalog of architectures crossed with training methods, and the two grow independently: a new
architecture works with every method of its family, a new method with every architecture. Most additions are one
class registered under one name, which then becomes a value you can write in a YAML. The patterns below are the
whole API; the shipped recipes are written the same way.
