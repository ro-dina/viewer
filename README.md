DICOM Viewer (2D + 3D)

医用画像（DICOM）を 2D / 3D で閲覧できる クロスプラットフォームアプリです。
2D ビューアは PySide6（Qt） ベース、3D ビューアは VTK を使用しています。

機能一覧
機能	説明
2D Viewer	Axial（横断）・Coronal（前額）・Sagittal（矢状）3面同時表示
WL/WW 調整	スライダーによるウィンドウレベル／ウィンドウ幅のリアルタイム変更
ガイド線表示	他の2面の交点位置をガイド線として重ね表示
3D Viewer	VTKベースのボリュームレンダリング（Composite / MIP / Band-pass対応）
マルチプラットフォーム	macOS (Universal), Windows 10/11 x64対応

ダウンロード
最新版は Releases ページから取得できます。

OS	ファイル名	備考
macOS (Apple Silicon / Intel)	DICOMViewer-macOS-arm64.zip	ダブルクリックで起動（初回はGatekeeper解除が必要）
Windows 10/11	DICOMViewer-Windows-x64.zip	解凍後、DICOMViewer2D.exe を実行
