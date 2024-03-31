"""Main App."""
import flet as ft

from pathlib import Path
from wildfire_detection.models import models_utils

PROJECT_ROOT = Path().resolve().parents[0]


def main(page: ft.Page) -> None:
    """Main function."""
    page.title = "Forest Fire Detection"
    page.scroll = "auto"
    page.theme_mode = ft.ThemeMode.DARK

    page.window_center()


    def run_predict(e: ft.ControlEvent) -> None:
        """Evaluate model and show image."""
        path_file = Path(image_holder.content.src)

        image_holder.content = ft.Container(
            ft.Column([
                ft.Text("Evaluating model..."),
                ft.ProgressBar(width=400, color="amber", bgcolor="#eeeeee"),
            ]),
            alignment=ft.alignment.bottom_center,
            padding=ft.padding.only(top=page.window_height / 2.5),
        )
        page.update()

        res = models_utils.evaluate_model(path_file)
        if res is True:
            predicted_data = PROJECT_ROOT / "data" / "predicted"
            predicted_data /= path_file.name

            image_holder.content = ft.Image(
                src=predicted_data,
                visible=True,
                fit=ft.ImageFit.CONTAIN,
            )
            page.update()


    def img_btn_clicked(e: ft.ControlEvent) -> None:
        """Show image."""
        image_holder.content.src = e.control.data
        image_holder.content.visible = True

        run_predictor_btn.color = "#FF6E1C"
        run_predictor_btn.bgcolor = "#8E0A3D"
        run_predictor_btn.disabled = False

        page.update()


    def container_item(lst) -> list[ft.Container]:
        """Add list of items to container."""
        names_lst = [path_file.name for path_file in lst]
        items = []

        for item, path in zip(names_lst, lst):
            items.append(
                ft.Container(
                    content=ft.TextButton(
                        text=str(item),
                        on_click=img_btn_clicked,
                        data=path,
                        icon="park_rounded",
                        icon_color="green600",
                    ),
                    alignment=ft.alignment.center_left,
                    bgcolor="#FFFFF",
                    border_radius=ft.border_radius.all(5),
                    padding=ft.padding.all(2),
                )
            )

        return items


    def handle_loaded_file(e: ft.FilePickerResultEvent) -> None:
        """File Picker."""
        files_dir = Path(e.files[0].path).resolve().parents[0]
        files_path = []

        # Get list files
        for ext in allowed_ext_files:
            files_path.extend(files_dir.glob(f"*.{ext}"))

        column_data.clean()
        column_data.content.controls = container_item(files_path)
        column_data.visible = True

        page.update()


    allowed_ext_files = ["jpg", "png", "jpeg"]
    select_image_btn = ft.ElevatedButton(
        text="Open Gallery",
        # bgcolor="green800",
        bgcolor="#019267",
        on_click=lambda _: file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=allowed_ext_files
        )
    )

    run_predictor_btn = ft.ElevatedButton(
        text="Run Prediction",
        on_click=run_predict,
        disabled=True,
    )

    buttons_row = ft.Container(
        ft.Row(
            controls=[select_image_btn, run_predictor_btn],
            spacing=25
        ),
    )


    file_picker = ft.FilePicker(on_result=handle_loaded_file)
    page.overlay.append(file_picker)

    column_data = ft.Container(
        content=ft.Column(
            spacing=10,
            height=100,
            width=100,
            scroll=ft.ScrollMode.ALWAYS,
        ),
        gradient=ft.LinearGradient(
            begin=ft.alignment.top_center,
            end=ft.alignment.bottom_center,
            colors=["#019267", "#4F4E4E"],
            rotation=45,
        ),
        width=300,
        height=page.window_height - 50,
        border_radius=ft.border_radius.all(5),
        padding=ft.padding.all(10),
        visible=False,
    )

    image_holder = ft.Container(
        content=ft.Image(
            visible=False,
            fit=ft.ImageFit.CONTAIN,
        ),
        width=page.window_width - column_data.padding.left - 90,
        height=page.window_height - 15,
        alignment=ft.alignment.center_left,
        border_radius=ft.border_radius.all(5),
    )

    row_container = ft.Container(
        content=ft.Row(
            controls=[column_data, image_holder],
            spacing=20,
        ),
    )

    page.add(buttons_row)
    page.add(row_container)

    page.update()


ft.app(target=main)
