from __future__ import annotations


class ClipPreflightMixin:
    def _ensure_status_steps_registered(self):
        registered_steps = set(self.status_manager.process_steps)
        required_registered_steps = set(self.required_steps)
        required_registered_steps.add(self.status_step)
        missing_steps = [step for step in required_registered_steps if step not in registered_steps]
        if missing_steps:
            raise ValueError(
                "配置中的 process_steps 缺少 clip 阶段所需步骤: "
                f"{', '.join(missing_steps)}"
            )

    def _ensure_folder_ready(self, folder_name: str, overwrite: bool) -> bool:
        if not self.store.folder_exists(folder_name):
            return False

        self.status_manager.ensure_status_file(folder_name)

        if not overwrite and self.status_manager.is_step_done(folder_name, self.status_step):
            return False

        missing_steps = [
            step for step in self.required_steps
            if not self.status_manager.is_step_done(folder_name, step)
        ]
        if missing_steps:
            raise ValueError(f"缺少前置步骤: {', '.join(missing_steps)}")

        return True
