"""Main App."""
import flet as ft
import folium
import webbrowser

from pathlib import Path

from wildfire_detection.models.models_utils import (
    evaluate_model,
    evaluate_model_video,
    open_web_camera_with_model,
)
from wildfire_detection.data.data_utils import (
    get_coords_location,
)

PROJECT_ROOT = Path().resolve().parents[0]


def main(page: ft.Page) -> None:
    """Main function."""
    # ----- Main Settings ----- #
    page.route = "/"
    page.title = "Forest Fire Detection"
    page.scroll = "auto"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_always_on_top = False
    page.window_center()

    # ----- Real Work Emulation ----- #
    # def loader_batch_images(
    #     stream_data: list[Path], batch_size: int = 4, delay_time: float = 1.0,
    # ):
    #     """Return batch images."""
    #     while True:
    #         data_available = len(stream_data) > 0
    #         if not data_available:
    #             break
    #
    #         batch_img = stream_data[:batch_size]
    #         stream_data = stream_data[batch_size:]
    #
    #         yield batch_img
    #         return

    def loading_real_data() -> None:
        """Loading Real Data."""
        data_dpath = PROJECT_ROOT / "data" / "raw" / "WildfireDataset" / "test" / "images"
        stream = list(data_dpath.glob("*"))

        # batch_loader = loader_batch_images(stream, batch_size=4)
        for img_fpath in stream:
            print(img_fpath)
            images_row.controls.append(
                ft.Image(
                    src=str(img_fpath),
                    width=200,
                    height=200,
                    fit=ft.ImageFit.CONTAIN,
                    visible=True,
                    # repeat=ft.ImageRepeat.NO_REPEAT,
                    # border_radius=ft.border_radius.all(10),
                ),
            )
        page.update()


    def run_real_work(e: ft.ControlEvent) -> None:
        """Create page for real working."""
        # page.clean()
        #
        # page.add(images_row)
        loading_real_data()

        page.update()


    images_row = ft.Row(
        expand=1,
        wrap=False,
        scroll=ft.ScrollMode.ALWAYS,
    )

    # ----- Functions ----- #
    def web_camera_clicked(e: ft.ControlEvent) -> None:
        """Open Web Camera with model evaluating."""
        open_web_camera_with_model()


    def map_btn_clicked(e: ft.ControlEvent) -> None:
        """Open map in browser."""
        current_img_fpath = image_holder.content.src
        img_fpath = current_img_fpath.resolve().parents[1] / "test" / current_img_fpath.name

        decimal_latitude, decimal_longitude = get_coords_location(img_fpath)
        url = f"https://www.google.com/maps?q={decimal_latitude},{decimal_longitude}"
        webbrowser.open_new_tab(url)

        # fol_map = folium.Map()
        #
        # folium.Marker(
        #     location=[6.243499,-75.579226],
        #     tooltip="Click for more information",
        #     popup="Medellin",
        # ).add_to(fol_map)
        #
        # fol_map.fit_bounds(fol_map.get_bounds())
        #
        # save_path = PROJECT_ROOT / "data" / "temp" / "map.html"
        # fol_map.save(save_path)
        #
        # webbrowser.open("file://" + str(save_path), new=2)


    def run_predict(e: ft.ControlEvent) -> None:
        """Evaluate model and show image."""
        path_file = Path(image_holder.content.src)
        if "mp4" in path_file.name:
            evaluate_model_video(path_file)
            return

        image_holder.content = ft.Container(
            ft.Column([
                ft.Text("Evaluating model..."),
                ft.ProgressBar(width=400, color="amber", bgcolor="#eeeeee"),
            ]),
            alignment=ft.alignment.bottom_center,
            padding=ft.padding.only(top=page.window_height / 2.5),
        )
        page.update()

        res = evaluate_model(path_file)
        if res is True:
            predicted_data = PROJECT_ROOT / "data" / "predicted"
            predicted_data /= path_file.name

            image_holder.content = ft.Image(
                src=predicted_data,
                visible=True,
                fit=ft.ImageFit.CONTAIN,
            )
            map_row.visible = True

            page.update()


    def img_btn_clicked(e: ft.ControlEvent) -> None:
        """Show image."""
        image_holder.content.src = e.control.data
        image_holder.content.visible = True

        run_predictor_btn.color = "#FF6E1C"
        run_predictor_btn.bgcolor = "#8E0A3D"
        run_predictor_btn.disabled = False

        page.update()


    def container_item(lst: list[Path]) -> list[ft.Container]:
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


    # ----- Top Row Of Buttons ----- #
    allowed_ext_files = ["jpg", "png", "jpeg", "mp4"]
    select_image_btn = ft.ElevatedButton(
        text="Open Gallery",
        bgcolor="#019267",
        on_click=lambda _: file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=allowed_ext_files,
        ),
    )

    run_predictor_btn = ft.ElevatedButton(
        text="Run Prediction",
        on_click=run_predict,
        disabled=True,
    )

    main_buttons = ft.Container(
        ft.Row(
            [select_image_btn, run_predictor_btn],
            spacing=20,
        ),
    )

    text_map = ft.Text("Open Map Location:")
    map_button = ft.IconButton(
        icon=ft.icons.MAP_SHARP,
        on_click=map_btn_clicked,
        icon_color="red600",
        icon_size=30,
        tooltip="Open Map",
    )

    map_row = ft.Container(
        ft.Row(
            [text_map, map_button],
            spacing=5,
        ),
        visible=False,
    )

    popup_menu_btn = ft.Container(
        ft.PopupMenuButton(
            items=[
                ft.PopupMenuItem(
                    icon=ft.icons.ARCHITECTURE,
                    text="Open Demonstrate Page",
                    on_click=None,
                ),
                ft.PopupMenuItem(),
                ft.PopupMenuItem(
                    icon=ft.icons.CAMERA,
                    text="Open Web Camera",
                    on_click=web_camera_clicked,
                ),
                ft.PopupMenuItem(),
                ft.PopupMenuItem(
                    icon=ft.icons.FOREST_SHARP,
                    text="Run Real Work",
                    on_click=run_real_work,
                ),
            ],
        ),
    )

    buttons_row = ft.Row(
        controls=[
            main_buttons,
            map_row,
            popup_menu_btn,
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )


    # ----- List And Image ----- #
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
        height=page.window_height - 70,
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


    # ----- Adding Elements ----- #
    page.add(buttons_row)
    page.add(row_container)

    page.update()


ft.app(target=main)
