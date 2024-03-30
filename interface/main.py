import base64
import flet as ft

from pathlib import Path


def main(page: ft.Page) -> None:
    """Main function."""
    page.title = "Images Example"
    page.scroll = "auto"
    page.theme_mode = ft.ThemeMode.LIGHT

    # page.window_width = 400
    # page.windows_height = 600

    # files_container = ft.Container(
    #     gradient=ft.LinearGradient(
    #         begin=ft.alignment.top_center,
    #         end=ft.alignment.bottom_center,
    #         colors=[ft.colors.BLUE, ft.colors.YELLOW],
    #     ),
    #
    #     width=150,
    #     height=150,
    #     border_radius=5,
    # )


    def handle_loaded_file(e: ft.FilePickerResultEvent):
        """File Picker."""
        files_dir = Path(e.files[0].path).resolve().parents[0]
        files_path = []

        # Get list files
        for ext in allowed_ext_files:
            files_path.extend(files_dir.glob(f"*.{ext}"))

        column_data.clean()

        column_data.content.controls = container_item([path_file.name for path_file in files_path])
        column_data.visible = True

        if e.files and len(e.files):
            with open(e.files[0].path, "rb") as r:
                image_holder.content.src_base64 = base64.b64encode(r.read()).decode("utf-8")
                image_holder.content.visible = True
                run_predictor_btn.disabled = False

        page.update()


    image_holder = ft.Container(
        ft.Image(
            visible=False,
            fit=ft.ImageFit.CONTAIN,
        ),
        bgcolor=ft.colors.GREY
    )

    file_picker = ft.FilePicker(on_result=handle_loaded_file)
    page.overlay.append(file_picker)

    column_data = ft.Container(
        content=ft.Column(
            spacing=10,
            height=100,
            width=float("inf"),
            scroll=ft.ScrollMode.ALWAYS,
            # controls=container_item(["Column"] * 10),
        ),
        width=400,
        height=500,
        bgcolor=ft.colors.INDIGO,
        border_radius=ft.border_radius.all(5),
        padding=ft.padding.all(10),
        visible=False,
    )

    row_container = ft.Container(
        ft.Row(
            controls=[column_data, image_holder],
            spacing=30,
        ),
    )

    allowed_ext_files = ["jpg", "png", "jpeg"]
    select_image_btn = ft.ElevatedButton(
        text="Select Image",
        on_click=lambda _: file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=allowed_ext_files
        )
    )

    run_predictor_btn = ft.ElevatedButton(
        text="Run Prediction",
        on_click=None,
        disabled=True,
    )

    buttons_row = ft.Container(
        ft.Row(
            controls=[select_image_btn, run_predictor_btn],
            spacing=30
        ),
        margin=ft.margin.only(bottom=10)
    )

    page.add(ft.Text("Read Local Image", size=30, color="green"))

    page.add(buttons_row)
    # page.add(image_holder)

    # page.add(column_data)
    page.add(row_container)

    page.update()


def container_item(lst):
    items = []
    for item in lst:
        items.append(
            ft.Container(
                content=ft.Text(value=str(item)),
                margin=ft.margin.only(left=5),
                alignment=ft.alignment.center,
                width=float("inf"),
                height=40,
                bgcolor=ft.colors.GREY,
                border_radius=ft.border_radius.all(5),
            )
        )

    return items


ft.app(target=main)
