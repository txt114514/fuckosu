from __future__ import annotations


class VerifyWrapUpMixin:
    def handle_failure(self, folder_name: str, error: Exception):
        if self.store.folder_exists(folder_name):
            self.status_manager.ensure_status_file(folder_name)
            self.status_manager.mark_step_pending(
                folder_name,
                "verify_exported",
                detail={"error": str(error)},
            )
