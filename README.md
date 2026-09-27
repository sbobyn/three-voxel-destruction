# three-voxel-destruction

A Teardown-inspired voxel destruction demo in Three.js (WebGPU): a city block of 94,000 voxels,
each a body in [three-avbd](https://github.com/sbobyn/three-avbd)'s GPU rigid-body solver. You
can hammer it, cut it with a laser, blow it up with rockets and charges, or throw a wrecking
ball. Storeys give way when what carries them is cut, glass shatters, and rubble settles and
freezes.

## Running it

```sh
pnpm install
pnpm dev        # http://localhost:5320
pnpm check      # typecheck, tests (GPU ones through Dawn), build
```

Needs a browser with WebGPU. The labs, `sky-lab.html` and `particles-lab.html`, tune the sky
and the particles on their own.

## three-avbd

It uses the solver through `three-avbd/advanced` (`GpuSolver3D` and the buffer layouts), since
it runs its own compute pass over the bodies (blasts). It depends on the published package
(`three-avbd` on npm). To work on both at once, link a checkout: `pnpm link ../three-avbd`
(after `pnpm build:lib` there).

The city is z-up (the solver's `up` is set to `REF_UP`).

## Credits

Textures and skies: [Poly Haven](https://polyhaven.com), CC0 (see `public/textures/CREDITS.md`
and `public/hdri/CREDITS.md`).
