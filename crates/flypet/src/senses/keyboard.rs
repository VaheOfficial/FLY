//! Keystrokes as a count, from a low-level keyboard hook.
//!
//! Only the number of key presses is recorded. Which key was pressed is
//! never read or stored: the fly feels the desk shake, it does not read.
//!
//! The hook must be installed on a thread that pumps window messages; the
//! winit event loop thread does.

use std::sync::atomic::{AtomicU32, Ordering};

use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, HC_ACTION, HHOOK, SetWindowsHookExW, UnhookWindowsHookEx, WH_KEYBOARD_LL,
    WM_KEYDOWN, WM_SYSKEYDOWN,
};

static KEYSTROKES: AtomicU32 = AtomicU32::new(0);

unsafe extern "system" fn count_key_presses(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if code == HC_ACTION as i32 {
        let message = wparam.0 as u32;
        if message == WM_KEYDOWN || message == WM_SYSKEYDOWN {
            KEYSTROKES.fetch_add(1, Ordering::Relaxed);
        }
    }
    unsafe { CallNextHookEx(None, code, wparam, lparam) }
}

/// A system-wide keystroke counter, removed on drop.
pub struct KeyboardHook {
    hook: HHOOK,
}

impl KeyboardHook {
    pub fn install() -> windows::core::Result<Self> {
        let hook = unsafe { SetWindowsHookExW(WH_KEYBOARD_LL, Some(count_key_presses), None, 0)? };
        Ok(Self { hook })
    }

    /// Key presses since the previous call.
    pub fn take_keystrokes(&self) -> u32 {
        KEYSTROKES.swap(0, Ordering::Relaxed)
    }
}

impl Drop for KeyboardHook {
    fn drop(&mut self) {
        // Failure here means the hook is already gone; nothing to do about it.
        let _ = unsafe { UnhookWindowsHookEx(self.hook) };
    }
}
