# Household props

glTF models that dress the tasks' objects (`look:` in a scene, see sim/fixtures.py).
sim/import_glb.py converts them to USD under external/home_assets/ on the instance.

| File | Model | Author | Licence | Used in |
|---|---|---|---|---|
| compressed_air_can.glb | [Compressed Air Can](https://sketchfab.com/3d-models/compressed-air-can-efb2ddcb7e13470d874b9b44bcd03048) | [PolyDavid](https://sketchfab.com/PolyDavid) | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | home_canister_hob |
| plastic_canister.glb | [#6 Plastic Canister (Lowpoly)](https://sketchfab.com/3d-models/6-plastic-canister-lowpoly-b87cd4d8f8b348e59daef66c6f0d3a1d) | [hoxsvl](https://sketchfab.com/hoxsvl) | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) | home_label_peel |
| rubber_duck_toy.glb | [Rubber Duck Toy](https://polyhaven.com/a/rubber_duck_toy) | [Poly Haven](https://polyhaven.com) | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | home_knife_toybox__realistic2 |
| baseball_01.glb | [Baseball 01](https://polyhaven.com/a/baseball_01) | [Poly Haven](https://polyhaven.com) | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | home_knife_toybox__realistic2 |

The first two are unchanged except for the second's file name. The Poly Haven ones are their glTF
with its 1k textures, packed into one glb. In the scenes they are scaled to fit the objects they dress.
