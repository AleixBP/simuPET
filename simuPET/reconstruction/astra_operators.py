from simuPET import array_lib as np
import astra

class parallel_projector(): #LinearOperator (super)
    def __init__(self, proj_geom, vol_geom, lip=None, eigen_min=None):
        self.proj_geom = proj_geom
        self.vol_geom = vol_geom
        self.P = astra.OpTomo(astra.creators.create_projector('line', proj_geom, vol_geom)) #'cuda'
        self.lip = np.infty if lip is None else lip
        self.eigen_min = 0 if eigen_min is None else eigen_min
        self.dim = [np.prod(np.asarray(astra.geom_size(vol_geom))), np.prod(np.asarray(astra.geom_size(proj_geom)))]

    def __call__(self, vol):
        return self.P*vol #comes out flattened


    def adjoint(self, sino):
        return (self.P).T*sino #comes out flattened


def init_parallel_projector_2D(spacing_of_s, number_of_s_samples, sampled_phi, vol_shp, **kwargs):
    proj_geom = astra.creators.create_proj_geom('parallel', \
                                                spacing_of_s, number_of_s_samples,
                                                sampled_phi)
    vol_geom = astra.creators.create_vol_geom(vol_shp)

    return parallel_projector(proj_geom, vol_geom, **kwargs)


class parallel_projector_CUPY(): #LinearOperator (super)

    def __init__(self, proj_geom, vol_geom, lip=None, eigen_min=None): #cuda3d #line
        self.proj_geom = proj_geom
        self.vol_geom = vol_geom
        self.data_mod = astra.data3d
        self.proj_id = astra.creators.create_projector("cuda3d", proj_geom, vol_geom) #cuda for 2d
        #self.proj_id = astra.creators.create_projector(mod, proj_geom, vol_geom)
        
        self.vol_shp = astra.geom_size(vol_geom)
        self.sino_shp = astra.geom_size(proj_geom)

        self.lip = np.infty if lip is None else lip
        self.eigen_min = 0 if eigen_min is None else eigen_min
        self.dim = [int(np.prod(np.asarray(astra.geom_size(vol_geom)))), int(np.prod(np.asarray(astra.geom_size(proj_geom))))]

    def check_array(self, arr, shp):
        if len(arr.shape)==1:
            arr = arr.reshape(shp)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.flags['C_CONTIGUOUS']==False:
            arr = np.ascontiguousarray(arr)
        # np.ascontiguousarray(vol, dtype=np.float32)
        return arr

    def FPBP(self, x, out=None, mode="FP"):

        if mode=="FP":
            vol = self.check_array(x, self.vol_shp)
            out = sino = np.empty(self.sino_shp, dtype=np.float32) if out is None else out
            vol_code = "VolumeDataId"
        else:
            sino = self.check_array(x, self.sino_shp)
            out = vol = np.empty(self.vol_shp, dtype=np.float32) if out is None else out
            vol_code = "ReconstructionDataId"

        vol_link = self.data_mod.GPULink(vol.data.ptr, *vol.shape[::-1], vol.strides[-2])
        vid = self.data_mod.link("-vol", self.vol_geom, vol_link)

        sino_link = self.data_mod.GPULink(sino.data.ptr, *sino.shape[::-1], sino.strides[-2])
        sid = self.data_mod.link('-sino', self.proj_geom, sino_link)

        cfg = astra.creators.astra_dict(mode+"3D_CUDA")
        cfg[vol_code] = vid
        cfg["ProjectionDataId"] = sid
        cfg['ProjectorId'] = self.proj_id #not sure necessary
        fp_id = astra.algorithm.create(cfg)
        astra.algorithm.run(fp_id)

        astra.algorithm.delete(fp_id)
        self.data_mod.delete([vid,sid])
        return out.ravel()

    def __call__(self, vol, out=None):
        return self.FPBP(vol, out=out, mode="FP")

    def adjoint(self, sino, out=None):
        return self.FPBP(sino, out=out, mode="BP")


#todo: (not sure) switch this: 1., spacing_of_s for spacing_of_s, 1.
def init_parallel_projector_CUPY_2D(spacing_of_s, number_of_s_samples, sampled_phi, vol_shp, **kwargs):
    proj_geom = astra.creators.create_proj_geom('parallel3d', \
                                                         1., spacing_of_s, \
                                                         1, number_of_s_samples, \
                                                         sampled_phi.get() \
                                                        )
    vol_geom = astra.creators.create_vol_geom(*vol_shp, 1)

    return parallel_projector_CUPY(proj_geom, vol_geom, **kwargs)


def fbp_reconstruction(proj_geom, vol_geom, parallel_sino_data):
    unif_id = astra.data2d.create('-sino', proj_geom, data=parallel_sino_data)
    alg_unif_cfg = astra.creators.astra_dict('FBP_CUDA') # BP_CUDA, SIRT_CUDA, SART_CUDA, CGLS_CUDA
    alg_unif_cfg['ProjectionDataId'] = unif_id
    rec_unif_id = astra.data2d.create('-vol', vol_geom, data=0) # empty to store result
    alg_unif_cfg['ReconstructionDataId'] = rec_unif_id
    alg_unif_id = astra.algorithm.create(alg_unif_cfg)
    astra.algorithm.run(alg_unif_id, iterations=1) # increase if not FBP (?)
    rec_unif = astra.data2d.get(rec_unif_id)
    return rec_unif


def bp_reconstruction(proj_geom, vol_geom, parallel_sino_data):
    shp = astra.geom_size(vol_geom)
    proj_id = astra.creators.create_projector('line', proj_geom, vol_geom)
    W = astra.OpTomo(proj_id)
    return (W.T*parallel_sino_data).reshape(shp)