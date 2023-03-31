use crate::allocator::Allocator;
use core::fmt;
use ncnn_bind::*;
use std::ops::{Index, IndexMut, RangeTo};
use std::os::raw::c_void;

pub enum MatPixelType {
    BGR,
    BGRA,
    GRAY,
    RGB,
    RGBA,
}

impl MatPixelType {
    pub fn from_int(i: i32) -> Option<Self> {
        match i as u32 {
            NCNN_MAT_PIXEL_BGR => Some(MatPixelType::BGR),
            NCNN_MAT_PIXEL_BGRA => Some(MatPixelType::BGRA),
            NCNN_MAT_PIXEL_GRAY => Some(MatPixelType::GRAY),
            NCNN_MAT_PIXEL_RGB => Some(MatPixelType::RGB),
            NCNN_MAT_PIXEL_RGBA => Some(MatPixelType::RGBA),
            _ => None,
        }
    }
    pub fn to_int(&self) -> i32 {
        match self {
            MatPixelType::BGR => NCNN_MAT_PIXEL_BGR as _,
            MatPixelType::BGRA => NCNN_MAT_PIXEL_BGRA as _,
            MatPixelType::GRAY => NCNN_MAT_PIXEL_GRAY as _,
            MatPixelType::RGB => NCNN_MAT_PIXEL_RGB as _,
            MatPixelType::RGBA => NCNN_MAT_PIXEL_RGBA as _,
        }
    }

    pub fn convert(&self, other: &Self) -> i32 {
        // const PIXEL_CONVERT_MASK: u32 = 0xffff0000;
        const PIXEL_CONVERT_SHIFT: usize = 16;
        self.to_int() | (other.to_int() << PIXEL_CONVERT_SHIFT)
    }

    pub fn stride(&self) -> i32 {
        match self {
            MatPixelType::BGR => 3,
            MatPixelType::BGRA => 4,
            MatPixelType::GRAY => 1,
            MatPixelType::RGB => 3,
            MatPixelType::RGBA => 4,
        }
    }
}

pub struct Mat {
    ptr: ncnn_mat_t,
}

// https://github.com/Tencent/ncnn/blob/5eb56b2ea5a99fb5a3d6f3669ef1743b73a9a53e/src/mat.h#L224
impl Mat {
    /// Constructs an empty matrix.
    pub fn new() -> Self {
        Self::default()
    }
    pub fn get_mut(&mut self) -> &mut ncnn_mat_t {
        &mut self.ptr
    }

    /// Constructs an empty 1D matrix.
    pub fn new_1d(w: i32, alloc: Option<&Allocator>) -> Self {
        Self {
            ptr: unsafe {
                ncnn_mat_create_1d(
                    w,
                    alloc.map(Allocator::ptr).unwrap_or(core::ptr::null_mut()),
                )
            },
        }
    }

    /// Constructs an empty 2D matrix.
    pub fn new_2d(w: i32, h: i32, alloc: Option<&Allocator>) -> Self {
        Self {
            ptr: unsafe {
                ncnn_mat_create_2d(
                    w,
                    h,
                    alloc.map(Allocator::ptr).unwrap_or(core::ptr::null_mut()),
                )
            },
        }
    }

    /// Constructs an empty 3D matrix.
    pub fn new_3d(w: i32, h: i32, c: i32, alloc: Option<&Allocator>) -> Self {
        Self {
            ptr: unsafe {
                ncnn_mat_create_3d(
                    w,
                    h,
                    c,
                    alloc.map(Allocator::ptr).unwrap_or(core::ptr::null_mut()),
                )
            },
        }
    }

    /// Constructs an empty 4D matrix.
    pub fn new_4d(w: i32, h: i32, d: i32, c: i32, alloc: Option<&Allocator>) -> Self {
        Self {
            ptr: unsafe {
                ncnn_mat_create_4d(
                    w,
                    h,
                    d,
                    c,
                    alloc.map(Allocator::ptr).unwrap_or(core::ptr::null_mut()),
                )
            },
        }
    }

    /// Constructs 1D matrix with a given raw data.
    ///
    /// # Safety
    ///
    /// Data pointer must not be aliased, it must be valid for the entire lifetime of Mat and it must be of correct size.
    pub unsafe fn new_external_1d(w: i32, data: *mut c_void, alloc: Option<&Allocator>) -> Self {
        Self {
            ptr: ncnn_mat_create_external_1d(
                w,
                data,
                alloc.map(Allocator::ptr).unwrap_or(core::ptr::null_mut()),
            ),
        }
    }

    /// Constructs 2D matrix with a given raw data.
    ///
    /// # Safety
    ///
    /// Data pointer must not be aliased, it must be valid for the entire lifetime of Mat and it must be of correct size.
    pub unsafe fn new_external_2d(
        w: i32,
        h: i32,
        data: *mut c_void,
        alloc: Option<&Allocator>,
    ) -> Self {
        Self {
            ptr: ncnn_mat_create_external_2d(
                w,
                h,
                data,
                alloc.map(Allocator::ptr).unwrap_or(core::ptr::null_mut()),
            ),
        }
    }

    /// Constructs 3D matrix with a given raw data.
    ///
    /// # Safety
    ///
    /// Data pointer must not be aliased, it must be valid for the entire lifetime of Mat and it must be of correct size.
    pub unsafe fn new_external_3d(
        w: i32,
        h: i32,
        c: i32,
        data: *mut c_void,
        alloc: Option<&Allocator>,
    ) -> Self {
        Self {
            ptr: ncnn_mat_create_external_3d(
                w,
                h,
                c,
                data,
                alloc.map(Allocator::ptr).unwrap_or(core::ptr::null_mut()),
            ),
        }
    }

    /// Constructs 4D matrix with a given raw data.
    ///
    /// # Safety
    ///
    /// Data pointer must not be aliased, it must be valid for the entire lifetime of Mat and it must be of correct size.
    pub unsafe fn new_external_4d(
        w: i32,
        h: i32,
        d: i32,
        c: i32,
        data: *mut c_void,
        alloc: Option<&Allocator>,
    ) -> Self {
        Self {
            ptr: ncnn_mat_create_external_4d(
                w,
                h,
                d,
                c,
                data,
                alloc.map(Allocator::ptr).unwrap_or(core::ptr::null_mut()),
            ),
        }
    }

    /// Constructs matrix from pixel byte array
    pub fn from_pixels(
        data: &[u8],
        pixel_type: MatPixelType,
        width: i32,
        height: i32,
        alloc: Option<&Allocator>,
    ) -> anyhow::Result<Mat> {
        let len = width * height * pixel_type.stride();
        if data.len() != len as _ {
            anyhow::bail!("Expected data length {}, provided {}", len, data.len());
        }

        Ok(Self {
            ptr: unsafe {
                ncnn_mat_from_pixels(
                    data.as_ptr(),
                    pixel_type.to_int(),
                    width,
                    height,
                    width * pixel_type.stride(),
                    alloc.map(Allocator::ptr).unwrap_or(core::ptr::null_mut()),
                )
            },
        })
    }

    // https://ncnn.docsforge.com/master/api/ncnn/Mat/from_pixels_resize/
    // https://ncnn.docsforge.com/master/api/ncnn_mat_from_pixels_resize/
    // https://github.com/Tencent/ncnn/blob/13a9533984467890a77acf5e26cc8d01ed157878/src/c_api.cpp#L365
    pub fn from_pixels_resize(
        data: &[u8],
        pixel_type: i32,
        input_size: (i32, i32),
        stride: i32,
        target_size: (i32, i32),
        alloc: Option<&Allocator>,
    ) -> anyhow::Result<Mat> {
        let (w, h) = input_size;
        let (model_w, model_h) = target_size;
        // let len = w * h * pixel_type.stride();
        // if data.len() != len as _ {
        //     anyhow::bail!("Expected data length {}, provided {}", len, data.len());
        // }
        unsafe {
            let ptr = ncnn_mat_from_pixels_resize(
                data.as_ptr(),
                pixel_type,
                w,
                h,
                stride,
                model_w,
                model_h,
                alloc.map(Allocator::ptr).unwrap_or(core::ptr::null_mut()),
            );
            Ok(Mat { ptr })
        }
    }

    pub fn substract_mean_normalize(&mut self, mean_vals: &[f32], norm_vals: &[f32]) {
        let channels = self.c() as usize;
        assert_eq!(mean_vals.len(), channels);
        assert_eq!(norm_vals.len(), channels);
        unsafe {
            ncnn_mat_substract_mean_normalize(self.ptr, mean_vals.as_ptr(), norm_vals.as_ptr())
        }
    }

    /// Fills matrix with a given value.
    pub fn fill(&mut self, value: f32) {
        unsafe { ncnn_mat_fill_float(self.ptr, value) };
    }

    /// Returns number of matrix dimensions.
    pub fn dims(&self) -> i32 {
        unsafe { ncnn_mat_get_dims(self.ptr) }
    }

    /// Returns matrix width
    pub fn w(&self) -> i32 {
        unsafe { ncnn_mat_get_w(self.ptr) }
    }

    /// Returns matrix height
    pub fn h(&self) -> i32 {
        unsafe { ncnn_mat_get_h(self.ptr) }
    }

    /// Returns matrix depth
    pub fn d(&self) -> i32 {
        unsafe { ncnn_mat_get_d(self.ptr) }
    }

    /// Returns matrix channels
    pub fn c(&self) -> i32 {
        unsafe { ncnn_mat_get_c(self.ptr) }
    }

    pub fn elemsize(&self) -> u64 {
        (unsafe { ncnn_mat_get_elemsize(self.ptr) }) as u64
    }

    pub fn elempack(&self) -> i32 {
        unsafe { ncnn_mat_get_elempack(self.ptr) }
    }

    pub fn cstep(&self) -> u64 {
        unsafe { ncnn_mat_get_cstep(self.ptr) as u64 }
    }

    /// Pointer to raw matrix data
    pub fn data(&self) -> *mut ::std::os::raw::c_void {
        unsafe { ncnn_mat_get_data(self.ptr) }
    }

    pub unsafe fn as_ptr(&self) -> ncnn_mat_t {
        self.ptr
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut ncnn_mat_t {
        &mut self.ptr
    }

    pub unsafe fn from_ptr(ptr: ncnn_mat_t) -> Self {
        Self { ptr: ptr }
    }

    /// The impl of C API has used `data` so you can't retrieve the Matrix.
    /// Note that the size of slice is unknown so don't trust `len()`
    ///
    /// See also [c_api.cpp](https://github.com/Tencent/ncnn/blob/a961ab992e2e4bf1cb950423bd7c2e2d40eb4ea2/src/c_api.cpp#LL375-L378C2).
    ///
    /// ```c
    /// void* ncnn_mat_get_channel_data(const ncnn_mat_t mat, int c){
    ///    return ((const Mat*)mat)->channel(c).data;
    /// }
    /// ```
    pub unsafe fn channel_data<T:Sized>(&self, c: i32) -> &[T] {
        let ptr = ncnn_mat_get_channel_data(self.ptr, c) as *mut T;
        let len = self.cstep();
        assert!(self.elemsize() as usize == std::mem::size_of::<T>());
        std::slice::from_raw_parts_mut(ptr, len as usize)
    }

    pub unsafe fn set_ptr(&mut self, ptr: ncnn_mat_t) {
        self.ptr = ptr;
    }

    pub fn isize_index_mut(&mut self, idx: isize) -> &mut f32 {
        let p = self.data() as *mut f32;
        unsafe {
            let p = p.offset(idx as isize);
            p.as_mut().unwrap()
        }
    }

    pub fn isize_index(&self, idx: isize) -> &f32 {
        let p = self.data() as *mut f32;
        unsafe {
            let p = p.offset(idx as isize);
            p.as_ref().unwrap()
        }
    }

    pub fn as_slice_mut<T:Sized>(&mut self) -> &mut [T] {
        let p = self.data() as *mut T;
        assert!(self.elemsize() % self.elempack() as u64 == 0);
        assert!(self.elemsize() as usize == std::mem::size_of::<T>());
        let len = self.elemsize() / self.elempack() as u64;
        unsafe { std::slice::from_raw_parts_mut(p, len as usize) }
    }

    pub fn as_slice<T:Sized>(&self) -> &[T] {
        let p = self.data() as *mut T;
        assert!(self.elemsize() % self.elempack() as u64 == 0);
        assert!(self.elemsize() as usize == std::mem::size_of::<T>());
        let len = self.elemsize() / self.elempack() as u64;
        unsafe { std::slice::from_raw_parts(p, len as usize) }
    }
}

impl Default for Mat {
    fn default() -> Self {
        Self {
            ptr: unsafe { ncnn_mat_create() },
        }
    }
}

impl fmt::Debug for Mat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mat")
            .field("dims", &self.dims())
            .field("c", &self.c())
            .field("h", &self.h())
            .field("w", &self.w())
            .field("elemsize", &self.elemsize())
            .field("elempack", &self.elempack())
            .field("cstep", &self.cstep())
            .finish()
    }
}

impl Drop for Mat {
    fn drop(&mut self) {
        unsafe {
            ncnn_mat_destroy(self.ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Mat;

    #[test]
    fn basic_getter_and_setter() {
        let m: Mat = Mat::new_3d(224, 224, 3, None);
        assert_eq!(224, m.h());
        assert_eq!(224, m.w());
        assert_eq!(3, m.c());
    }
}
