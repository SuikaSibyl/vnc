(() => {
  const { createApp } = Vue;

  const cases = [
    {
      id: "house",
      name: "House",
      thumb: "resources/pose/house/gt_nv.png",
      description: "Pose case with init and target images plus six method videos.",
      initImage: "resources/pose/house/init.png",
      targetImage: "resources/pose/house/gt_nv.png",
      category: "Pose",
      methodCount: 6,
      matrixColumns: [
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init image", kind: "image", src: "resources/pose/house/init.png" },
            { id: "target", label: "Target image", kind: "image", src: "resources/pose/house/gt_nv.png" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 0,
          cells: [
            { id: "nvd", label: "Nvdiffrast", kind: "video", video: "resources/pose/house/nvd.mp4" },
            { id: "p3d_sil", label: "Pytorch3D", kind: "video", video: "resources/pose/house/p3d_sil.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "rgbxy", label: "RGBXY", kind: "video", video: "resources/pose/house/rgbxy.mp4" },
            { id: "prdpt", label: "PRDPT", kind: "video", video: "resources/pose/house/prdpt.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 4,
          cells: [
            { id: "generated_ref", label: "Generated Video", kind: "video", video: "resources/pose/house/generated_ref.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/pose/house/vnc.mp4" },
          ],
        },
      ],
    },
    {
      id: "castle",
      name: "Castle",
      thumb: "resources/pose/castle/gt_nv.png",
      previewVideo: null,
      category: "Pose",
      description: "Pose estimation comparison with six method videos.",
      methodCount: 6,
      matrixColumns: [
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init image", kind: "image", src: "resources/pose/castle/init.png" },
            { id: "target", label: "Target image", kind: "image", src: "resources/pose/castle/gt_nv.png" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 0,
          cells: [
            { id: "nvdiffrast", label: "Nvdiffrast", kind: "video", video: "resources/pose/castle/nvdiffrast.mp4" },
            { id: "p3d", label: "Pytorch3D", kind: "video", video: "resources/pose/castle/p3d.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "rgbxy", label: "RGBXY", kind: "video", video: "resources/pose/castle/rgbxy.mp4" },
            { id: "prdpt", label: "PRDPT", kind: "video", video: "resources/pose/castle/prdpt.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 4,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/pose/castle/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/pose/castle/vnc.mp4" },
          ],
        },
      ],
    },
    {
      id: "mirror",
      name: "Mirror",
      thumb: "resources/pose/mirror/reference.png",
      category: "Pose",
      description: "Pose estimation comparison with six method videos.",
      methodCount: 6,
      matrixColumns: [
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init image", kind: "image", src: "resources/pose/mirror/initial.png" },
            { id: "target", label: "Target image", kind: "image", src: "resources/pose/mirror/reference.png" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 0,
          cells: [
            { id: "mse", label: "MSE", kind: "video", video: "resources/pose/mirror/mse.mp4" },
            { id: "loir", label: "LOIR", kind: "video", video: "resources/pose/mirror/loir.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "rgbxy", label: "RGBXY", kind: "na" },
            { id: "prdpt", label: "PRDPT", kind: "video", video: "resources/pose/mirror/prdpt.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 4,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/pose/mirror/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/pose/mirror/vnc.mp4" },
          ],
        },
      ],
    },
    {
      id: "penguin",
      name: "Penguin",
      thumb: "resources/pose/penguin/gt_nv.png",
      category: "Pose",
      description: "Pose estimation comparison with six method videos.",
      methodCount: 6,
      matrixColumns: [
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init image", kind: "image", src: "resources/pose/penguin/init.png" },
            { id: "target", label: "Target image", kind: "image", src: "resources/pose/penguin/gt_nv.png" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 0,
          cells: [
            { id: "nvdiffrast", label: "Nvdiffrast", kind: "video", video: "resources/pose/penguin/nvdiffrast.mp4" },
            { id: "p3d", label: "Pytorch3D", kind: "video", video: "resources/pose/penguin/p3d.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "rgbxy", label: "RGBXY", kind: "video", video: "resources/pose/penguin/rgbxy.mp4" },
            { id: "prdpt", label: "PRDPT", kind: "video", video: "resources/pose/penguin/prdpt.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 4,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/pose/penguin/generated_video.mp4" },
            { id: "vhc", label: "VHC (ours)", kind: "video", video: "resources/pose/penguin/vhc.mp4" },
          ],
        },
      ],
    },
    {
      id: "furniture",
      name: "Furniture",
      thumb: "resources/pose/furniture/ref.png",
      category: "Pose",
      description: "Pose estimation comparison with six method videos.",
      methodCount: 6,
      matrixColumns: [
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init image", kind: "image", src: "resources/pose/furniture/init.png" },
            { id: "target", label: "Target image", kind: "image", src: "resources/pose/furniture/ref.png" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 0,
          cells: [
            { id: "nvdiffrast", label: "Nvdiffrast", kind: "video", video: "resources/pose/furniture/nvdiffrast.mp4" },
            { id: "loir", label: "LOIR", kind: "video", video: "resources/pose/furniture/loir.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "rgbxy", label: "RGBXY", kind: "video", video: "resources/pose/furniture/rgbxy.mp4" },
            { id: "prdpt", label: "PRDPT", kind: "video", video: "resources/pose/furniture/prdpt.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 4,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/pose/furniture/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/pose/furniture/vnc.mp4" },
          ],
        },
      ],
    },
    {
      id: "rubiks",
      name: "Rubik's Cube",
      thumb: "resources/pose/rubiks/gt_nv.png",
      category: "Pose",
      description: "",
      methodCount: 6,
      matrixColumns: [
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init image", kind: "image", src: "resources/pose/rubiks/init.png" },
            { id: "target", label: "Target image", kind: "image", src: "resources/pose/rubiks/gt_nv.png" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 0,
          cells: [
            { id: "nvdiffrast", label: "Nvdiffrast", kind: "video", video: "resources/pose/rubiks/nvdiffrast.mp4" },
            { id: "p3d", label: "Pytorch3D", kind: "video", video: "resources/pose/rubiks/p3d.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "rgbxy", label: "RGBXY", kind: "video", video: "resources/pose/rubiks/rgbxy.mp4" },
            { id: "prdpt", label: "PRDPT", kind: "video", video: "resources/pose/rubiks/prdpt.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 4,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/pose/rubiks/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/pose/rubiks/vnc.mp4" },
          ],
        },
      ],
    },
    {
      id: "mug",
      name: "Mug",
      thumb: "resources/pose/mug/gt_nv.png",
      category: "Pose",
      description: "",
      methodCount: 4,
      matrixColumns: [
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init image", kind: "image", src: "resources/pose/mug/init.png" },
            { id: "target", label: "Target image", kind: "image", src: "resources/pose/mug/gt_nv.png" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 0,
          cells: [
            { id: "nvdiffrast", label: "Nvdiffrast", kind: "video", video: "resources/pose/mug/nvdiffrast.mp4" },
            { id: "prdpt", label: "PRDPT", kind: "video", video: "resources/pose/mug/prdpt.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/pose/mug/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/pose/mug/vnc.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 4,
          cells: [
            { id: "generated_video_copy", label: "Generated Video", kind: "video", video: "resources/pose/mug/generated_video_copy.mp4" },
            { id: "vnc_prdpt", label: "VNC + PRDPT (ours)", kind: "video", video: "resources/pose/mug/vnc+prdpt.mp4" },
          ],
        },
      ],
    },
    {
      id: "gnome",
      name: "Gnome",
      thumb: "resources/pose/gnome/gt_raw.png",
      category: "Pose",
      description: "",
      methodCount: 3,
      matrixColumns: [
        {
          title: "Init",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init", kind: "image", src: "resources/pose/gnome/init.png" },
          ],
        },
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "target", label: "Target", kind: "image", src: "resources/pose/gnome/gt_raw.png" },
            { id: "mse", label: "MSE", kind: "video", video: "resources/pose/gnome/mse.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/pose/gnome/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/pose/gnome/vnc.mp4" },
          ],
        },
      ],
    },
    {
      id: "earth",
      name: "Earth",
      thumb: "resources/pose/earth/gt.png",
      category: "Pose",
      description: "",
      methodCount: 6,
      matrixColumns: [
        {
          title: "Init",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init", kind: "image", src: "resources/pose/earth/init.png" },
          ],
        },
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "target", label: "Target", kind: "image", src: "resources/pose/earth/gt.png" },
            { id: "earth_ref", label: "Reference Video", kind: "video", video: "resources/pose/earth/earth.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "nvdiffrast", label: "Nvdiffrast", kind: "video", video: "resources/pose/earth/nvdiffrast.mp4" },
            { id: "loir", label: "LOIR", kind: "video", video: "resources/pose/earth/loir.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 4,
          cells: [
            { id: "rgbxy", label: "RGBXY", kind: "video", video: "resources/pose/earth/rgbxy.mp4" },
            { id: "prdpt", label: "PRDPT", kind: "video", video: "resources/pose/earth/prdpt.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 6,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/pose/earth/generated-video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/pose/earth/vnc.mp4" },
          ],
        },
      ],
    },
    {
      id: "breakdance",
      name: "Breakdance",
      thumb: "resources/skeleton/breakdance/target.png",
      category: "Skeleton",
      description: "",
      methodCount: 6,
      matrixColumns: [
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init image", kind: "image", src: "resources/skeleton/breakdance/init.png" },
            { id: "target", label: "Target image", kind: "image", src: "resources/skeleton/breakdance/target.png" },
          ],
        },
        {
          title: "Baselines",
          videoOffset: 0,
          cells: [
            { id: "mse", label: "MSE", kind: "video", video: "resources/skeleton/breakdance/mse.mp4" },
            { id: "loir", label: "LOIR", kind: "video", video: "resources/skeleton/breakdance/loir.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "rgbxy", label: "RGBXY", kind: "video", video: "resources/skeleton/breakdance/rgbxy.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/skeleton/breakdance/vnc.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 4,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/skeleton/breakdance/generated_video.mp4" },
            { id: "vnc_rgbxy", label: "VNC + RGBXY (ours)", kind: "video", video: "resources/skeleton/breakdance/vnc+rgbxy.mp4" },
          ],
        },
      ],
    },
    {
      id: "uprock",
      name: "Uprock",
      thumb: "resources/skeleton/uprock/target.png",
      category: "Skeleton",
      description: "",
      methodCount: 6,
      matrixColumns: [
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init image", kind: "image", src: "resources/skeleton/uprock/init.png" },
            { id: "target", label: "Target image", kind: "image", src: "resources/skeleton/uprock/target.png" },
          ],
        },
        {
          title: "Baselines",
          videoOffset: 0,
          cells: [
            { id: "mse", label: "MSE", kind: "video", video: "resources/skeleton/uprock/mse.mp4" },
            { id: "loir", label: "LOIR", kind: "video", video: "resources/skeleton/uprock/loir.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "rgbxy", label: "RGBXY", kind: "video", video: "resources/skeleton/uprock/rgbxy.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/skeleton/uprock/vnc.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 4,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/skeleton/uprock/generated_video.mp4" },
            { id: "vnc_rgbxy", label: "VNC + RGBXY (ours)", kind: "video", video: "resources/skeleton/uprock/vnc+rgbxy.mp4" },
          ],
        },
      ],
    },
    {
      id: "floor",
      name: "Floor",
      thumb: "resources/lights/floor/target.png",
      category: "Lights",
      description: "",
      methodCount: 6,
      matrixColumns: [
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init image", kind: "image", src: "resources/lights/floor/init.png" },
            { id: "target", label: "Target image", kind: "image", src: "resources/lights/floor/target.png" },
          ],
        },
        {
          title: "Baselines",
          videoOffset: 0,
          cells: [
            { id: "mse", label: "MSE", kind: "video", video: "resources/lights/floor/mse.mp4" },
            { id: "loir", label: "LOIR", kind: "video", video: "resources/lights/floor/loir.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "rgbxy", label: "RGBXY", kind: "video", video: "resources/lights/floor/rgbxy.mp4" },
            { id: "prdpt", label: "PRDPT", kind: "video", video: "resources/lights/floor/prdpt.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 4,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/lights/floor/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/lights/floor/vnc.mp4" },
          ],
        },
      ],
    },
    {
      id: "shadow",
      name: "Shadow",
      thumb: "resources/lights/shadow/target.png",
      category: "Lights",
      description: "",
      methodCount: 5,
      matrixColumns: [
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init image", kind: "image", src: "resources/lights/shadow/init.png" },
            { id: "target", label: "Target image", kind: "image", src: "resources/lights/shadow/target.png" },
          ],
        },
        {
          title: "Baselines",
          videoOffset: 0,
          cells: [
            { id: "mse", label: "MSE", kind: "video", video: "resources/lights/shadow/mse.mp4" },
            { id: "loir", label: "LOIR", kind: "video", video: "resources/lights/shadow/loir.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "rgbxy", label: "RGBXY", kind: "na" },
            { id: "prdpt", label: "PRDPT", kind: "video", video: "resources/lights/shadow/prdpt.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 4,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/lights/shadow/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/lights/shadow/vnc.mp4" },
          ],
        },
      ],
    },
    {
      id: "curve",
      name: "Curve",
      thumb: "resources/geometry/curve/target.png",
      category: "Geometry",
      description: "",
      methodCount: 3,
      freeAspect: true,
      matrixColumns: [
        {
          title: "Init",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init", kind: "image", src: "resources/geometry/curve/init.png" },
          ],
        },
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "target", label: "Target", kind: "image", src: "resources/geometry/curve/target.png" },
            { id: "mse", label: "MSE", kind: "video", video: "resources/geometry/curve/mse.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/geometry/curve/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/geometry/curve/vnc.mp4" },
          ],
        },
        {
          title: "Full View",
          videoOffset: 4,
          cells: [
            { id: "mse_fullview", label: "MSE Full View", kind: "video", video: "resources/geometry/curve/mse_fullview.mp4" },
            { id: "vnc_fullview", label: "VNC Full View (ours)", kind: "video", video: "resources/geometry/curve/vnc_fullview.mp4" },
          ],
        },
      ],
    },
    {
      id: "heart",
      name: "Heart",
      thumb: "resources/geometry/heart/ref_view_000.png",
      category: "Geometry",
      description: "Geometry case with init, target, and method comparison.",
      methodCount: 3,
      freeAspect: true,
      matrixColumns: [
        {
          title: "Init",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init", kind: "image", src: "resources/geometry/heart/init_view_000.png" },
          ],
        },
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "target", label: "Target", kind: "image", src: "resources/geometry/heart/ref_view_000.png" },
            { id: "largestep", label: "Largestep", kind: "video", video: "resources/geometry/heart/largestep.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/geometry/heart/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/geometry/heart/vnc.mp4" },
          ],
        },
      ],
    },
    {
      id: "bob",
      name: "Bob",
      thumb: "resources/geometry/bob/ref_view_000.png",
      category: "Geometry",
      description: "Geometry case with init, target, and method comparison.",
      methodCount: 3,
      freeAspect: true,
      matrixColumns: [
        {
          title: "Init",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init", kind: "image", src: "resources/geometry/bob/init_view_000.png" },
          ],
        },
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "target", label: "Target", kind: "image", src: "resources/geometry/bob/ref_view_000.png" },
            { id: "mse", label: "MSE", kind: "video", video: "resources/geometry/bob/mse.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/geometry/bob/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/geometry/bob/vnc.mp4" },
          ],
        },
      ],
    },
    {
      id: "bunny",
      name: "Bunny",
      thumb: "resources/geometry/bunny/ref_view_005.png",
      category: "Geometry",
      description: "Geometry case with two generated and VNC output pairs.",
      methodCount: 3,
      freeAspect: true,
      matrixColumns: [
        {
          title: "Init",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init", kind: "image", src: "resources/geometry/bunny/init_view_005.png" },
          ],
        },
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "target", label: "Target", kind: "image", src: "resources/geometry/bunny/ref_view_005.png" },
            { id: "mse", label: "MSE", kind: "video", video: "resources/geometry/bunny/mse.mp4" },
          ],
        },
        {
          title: "Pair 1",
          videoOffset: 2,
          cells: [
            { id: "generated_video1", label: "Generated Video 1", kind: "video", video: "resources/geometry/bunny/generated_video1.mp4" },
            { id: "vnc1", label: "VNC 1 (ours)", kind: "video", video: "resources/geometry/bunny/vnc_video1.mp4" },
          ],
        },
        {
          title: "Pair 2",
          videoOffset: 4,
          cells: [
            { id: "generated_video2", label: "Generated Video 2", kind: "video", video: "resources/geometry/bunny/generated_video2.mp4" },
            { id: "vnc2", label: "VNC 2 (ours)", kind: "video", video: "resources/geometry/bunny/vnc_video2.mp4" },
          ],
        },
      ],
    },
    {
      id: "suzanne",
      name: "Suzanne",
      category: "Geometry",
      thumb: "resources/geometry/suzanne/ref_view_001.png",
      description: "Geometry case with three scenario comparisons.",
      methodCount: 3,
      freeAspect: true,
      matrixColumns: [
        {
          title: "Init",
          videoOffset: 0,
          cells: [
            { id: "init1", label: "Init 1", kind: "image", src: "resources/geometry/suzanne/init_view_001.png" },
            { id: "init2", label: "Init 2", kind: "image", src: "resources/geometry/suzanne/init_view_002.png" },
            { id: "init3", label: "Init 3", kind: "image", src: "resources/geometry/suzanne/init_view_003.png" },
          ],
        },
        {
          title: "Target",
          videoOffset: 0,
          cells: [
            { id: "ref1", label: "Target 1", kind: "image", src: "resources/geometry/suzanne/ref_view_001.png" },
            { id: "ref2", label: "Target 2", kind: "image", src: "resources/geometry/suzanne/ref_view_005.png" },
            { id: "ref3", label: "Target 3", kind: "image", src: "resources/geometry/suzanne/ref_view_009.png" },
          ],
        },
        {
          title: "MSE",
          videoOffset: 0,
          cells: [
            { id: "mse1", label: "MSE 1", kind: "video", video: "resources/geometry/suzanne/mse_001.mp4" },
            { id: "mse2", label: "MSE 2", kind: "video", video: "resources/geometry/suzanne/mse_002.mp4" },
            { id: "mse3", label: "MSE 3", kind: "video", video: "resources/geometry/suzanne/mse_003.mp4" },
          ],
        },
        {
          title: "Generated Video",
          videoOffset: 3,
          cells: [
            { id: "gen1", label: "Generated Video 1", kind: "video", video: "resources/geometry/suzanne/generated_video_001.mp4" },
            { id: "gen2", label: "Generated Video 2", kind: "video", video: "resources/geometry/suzanne/generated_video_002.mp4" },
            { id: "gen3", label: "Generated Video 3", kind: "video", video: "resources/geometry/suzanne/generated_video_003.mp4" },
          ],
        },
        {
          title: "VNC (ours)",
          videoOffset: 6,
          cells: [
            { id: "vnc1", label: "VNC 1 (ours)", kind: "video", video: "resources/geometry/suzanne/vnc_001.mp4" },
            { id: "vnc2", label: "VNC 2 (ours)", kind: "video", video: "resources/geometry/suzanne/vnc_002.mp4" },
            { id: "vnc3", label: "VNC 3 (ours)", kind: "video", video: "resources/geometry/suzanne/vnc_003.mp4" },
          ],
        },
      ],
    },
    {
      id: "dragon",
      name: "Dragon",
      category: "Geometry",
      thumb: "resources/geometry/dragon/ref_view_019.png",
      description: "Geometry case with init, target, and method comparison.",
      methodCount: 3,
      freeAspect: true,
      matrixColumns: [
        {
          title: "Init",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init", kind: "image", src: "resources/geometry/dragon/init_view_019.png" },
          ],
        },
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "target", label: "Target", kind: "image", src: "resources/geometry/dragon/ref_view_019.png" },
            { id: "mse", label: "MSE", kind: "video", video: "resources/geometry/dragon/mse.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/geometry/dragon/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/geometry/dragon/vnc.mp4" },
          ],
        },
      ],
    },
    {
      id: "teapot",
      name: "Teapot",
      thumb: "resources/material/teapot/ref_ldr.png",
      category: "Material",
      description: "Material case with init, target, and method comparison.",
      methodCount: 3,
      freeAspect: true,
      matrixColumns: [
        {
          title: "Init",
          videoOffset: 0,
          cells: [
            { id: "init", label: "Init", kind: "image", src: "resources/material/teapot/init_ldr.png" },
          ],
        },
        {
          title: "Reference",
          videoOffset: 0,
          cells: [
            { id: "target", label: "Target", kind: "image", src: "resources/material/teapot/ref_ldr.png" },
            { id: "mse", label: "MSE", kind: "video", video: "resources/material/teapot/mse.mp4" },
          ],
        },
        {
          title: "Methods",
          videoOffset: 2,
          cells: [
            { id: "generated_video", label: "Generated Video", kind: "video", video: "resources/material/teapot/generated_video.mp4" },
            { id: "vnc", label: "VNC (ours)", kind: "video", video: "resources/material/teapot/vnc.mp4" },
          ],
        },
      ],
    },
  ];

  const page = document.body.dataset.page || "index";
  const activeCaseId = new URLSearchParams(window.location.search).get("case") || cases[0].id;

  const app = createApp({
    data() {
      return {
        cases,
        page,
        activeCaseId,
        videoRefs: [],
        cleanupSync: () => {},
      };
    },
    computed: {
      isIndex() {
        return this.page === "index";
      },
      isCase() {
        return this.page === "case";
      },
      activeCase() {
        return this.cases.find((item) => item.id === this.activeCaseId) || this.cases[0];
      },
      methodCount() {
        return this.activeCase.methodCount;
      },
    },
    methods: {
      caseLink(caseItem) {
        return `case.html?case=${encodeURIComponent(caseItem.id)}`;
      },
      openCase(caseItem) {
        window.location.href = this.caseLink(caseItem);
      },
      goHome() {
        window.location.href = "index.html";
      },
      setVideoRef(element, index) {
        if (element) this.videoRefs[index] = element;
      },
      startSync() {
        this.cleanupSync();

        const videos = this.videoRefs.filter(Boolean);
        if (videos.length === 0) {
          return;
        }

        const master = videos[0];
        let rafId = 0;

        const syncFollowers = () => {
          const currentTime = master.currentTime;
          for (const video of videos.slice(1)) {
            if (Math.abs(video.currentTime - currentTime) > 0.08) {
              video.currentTime = currentTime;
            }
            if (video.paused && !master.paused) {
              video.play().catch(() => {});
            }
          }
        };

        const tick = () => {
          if (master.paused) {
            return;
          }
          syncFollowers();
          rafId = window.requestAnimationFrame(tick);
        };

        const onSeeked = () => syncFollowers();

        for (const video of videos) {
          video.loop = true;
          video.muted = true;
          video.playsInline = true;
          video.addEventListener("seeked", onSeeked);
        }

        Promise.all(videos.map((video) => video.play().catch(() => {}))).then(() => {
          syncFollowers();
          rafId = window.requestAnimationFrame(tick);
        });

        this.cleanupSync = () => {
          if (rafId) {
            window.cancelAnimationFrame(rafId);
          }
          for (const video of videos) {
            video.removeEventListener("seeked", onSeeked);
          }
        };
      },
    },
    mounted() {
      if (this.isCase) {
        this.$nextTick(() => this.startSync());
      }
    },
    beforeUnmount() {
      this.cleanupSync();
    },
    template: `
      <main class="page-shell" :class="{ 'page-shell--case': isCase }">
        <section v-if="isIndex" class="hero-stage">
          <div class="index-hero">
            <div>
              <h1>Cases</h1>
              <p class="lede">Synchronized multi-method comparison results across pose, geometry, and material tasks.</p>
            </div>
            <el-tag type="success" effect="light" size="large">{{ cases.length }} cases</el-tag>
          </div>

          <div class="cases-grid">
            <el-card v-for="caseItem in cases" :key="caseItem.id" shadow="hover" class="case-card" @click="openCase(caseItem)">
              <template #header>
                <div class="card-head">
                  <div>
                    <h2>{{ caseItem.name }}</h2>
                  </div>
                  <el-tag size="small">{{ caseItem.category }}</el-tag>
                </div>
              </template>

              <el-image v-if="caseItem.thumb" class="case-thumb" :src="caseItem.thumb" fit="cover" lazy />
              <video v-else-if="caseItem.previewVideo" class="case-thumb case-thumb--video" :src="caseItem.previewVideo" autoplay muted loop playsinline></video>
              <div v-else class="case-thumb case-thumb--empty"><span>No preview</span></div>

              <div class="case-card__footer">
                <el-button type="primary" @click.stop="openCase(caseItem)">Open case</el-button>
              </div>
            </el-card>
          </div>
        </section>

        <section v-else class="case-stage" :class="{ 'case-stage--free-aspect': activeCase.freeAspect }">
          <el-page-header class="case-header" @back="goHome">
            <template #content>
              <div class="case-header__content">
                <div>
                  <div class="case-header__eyebrow">Case detail</div>
                  <div class="case-header__title">{{ activeCase.name }}</div>
                  <div class="case-header__subtitle">{{ activeCase.description }}</div>
                </div>
              </div>
            </template>
          </el-page-header>

          <div class="matrix-grid" :style="{ '--col-count': activeCase.matrixColumns.length }">
            <article v-for="(column, columnIndex) in activeCase.matrixColumns" :key="column.title + columnIndex" class="matrix-column">
              <div class="matrix-column__cells">
                <article v-for="(cell, cellIndex) in column.cells" :key="cell.id" class="matrix-cell" :class="['matrix-cell--' + cell.kind, cell.label.toLowerCase().includes('ours') ? 'matrix-cell--featured' : '']">
                  <div class="media-title">
                    <span>{{ cell.label }}</span>
                    <el-tag size="small">{{ cell.kind === 'image' ? 'PNG' : cell.kind === 'video' ? 'MP4' : 'N/A' }}</el-tag>
                  </div>
                  <div class="matrix-frame" :class="'matrix-frame--' + cell.kind">
                    <el-image
                      v-if="cell.kind === 'image'"
                      :src="cell.src"
                      :alt="activeCase.name + ' ' + cell.label"
                      :fit="activeCase.freeAspect ? 'contain' : 'cover'"
                    />

                    <div v-else-if="cell.kind === 'na'" class="na-placeholder"></div>

                    <video
                      v-else
                      :ref="(element) => setVideoRef(element, column.videoOffset + cellIndex)"
                      class="sync-video"
                      :src="cell.video"
                      autoplay
                      muted
                      loop
                      playsinline
                      preload="auto"
                      controls
                    ></video>
                  </div>
                </article>
              </div>
            </article>
          </div>
        </section>
      </main>
    `,
  });

  app.use(ElementPlus);
  app.mount("#app");
})();