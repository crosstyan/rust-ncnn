use crate::allocator::Allocator;
use core::fmt;
use ncnn_bind::*;
use std::ops::{Index, IndexMut};
use std::os::raw::c_void;

// like a mat but don't drop
pub struct MatView {
    ptr: ncnn_mat_t,
    // _phantom: PhantomData<&'a ()>,
}

impl MatView {
    pub unsafe fn from_ptr(ptr: ncnn_mat_t) -> Self {
        Self { ptr: ptr }
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

    pub unsafe fn set_ptr(&mut self, ptr: ncnn_mat_t) {
        self.ptr = ptr;
    }
}

// TODO: generic indexing?
// TODO: index return value?
impl Index<isize> for MatView {
    type Output = f32;
    // https://github.com/Tencent/ncnn/blob/5eb56b2ea5a99fb5a3d6f3669ef1743b73a9a53e/src/mat.h#L1343
    // https://stackoverflow.com/questions/24759028/how-should-you-do-pointer-arithmetic-in-rust
    fn index(&self, idx: isize) -> &Self::Output {
        let p = self.data() as *mut f32;
        unsafe {
            let p = p.offset(idx);
            p.as_ref().unwrap()
        }
    }
}

impl IndexMut<isize> for MatView {
    fn index_mut(&mut self, idx: isize) -> &mut Self::Output {
        let p = self.data() as *mut f32;
        unsafe {
            let p = p.offset(idx);
            p.as_mut().unwrap()
        }
    }
}
