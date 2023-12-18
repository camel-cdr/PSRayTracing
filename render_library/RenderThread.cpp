#include "RenderThread.hpp"
#include "Interfaces/IHittable.hpp"
#include "Interfaces/ICamera.hpp"
#include "Interfaces/IMaterial.hpp"
#include "Ray.hpp"
#include "Util.hpp"
#include "HitRecord.hpp"
#include "ScatterRecord.hpp"
#include "RenderOutput.hpp"
#include "RandomGenerator.hpp"
#include "PDFVariant.hpp"
#include "PDFs/CosinePDF.hpp"
#include "PDFs/HittablePDF.hpp"
#include "PDFs/MixturePDF.hpp"
using namespace std;



/** The funtion where the rendering magic happens. */
Vec3 ray_color(const RenderContext &r_ctx, RandomGenerator &rng, const Ray &r, const uint32_t depth) NOEXCEPT;

/** Finds what the color should be for the given pixel */
ColorRGBA _pixel_color(const RenderContext &r_ctx, RandomGenerator &rng, const uint32_t samples_per_pixel, const rreal x, const rreal y, const uint16_t max_ray_depth) NOEXCEPT;

/** Resolves a PDFVariant to a pointer to an IPDF */
const IPDF *maybe_get_pdf_ptr(const PDFVariant &pdf) NOEXCEPT;



RenderThread::RenderThread(const RenderContext &render_context) NOEXCEPT :
    _r_ctx(render_context)
{
    _running.store(false);
    _stop_requested.store(false);
    _completed_task_count.store(0);

    _total_queued_pixels.store(0);
    _pixels_completed.store(0);
}

RenderThread::RenderThread(RenderThread &&other) NOEXCEPT :
    _r_ctx(move(other._r_ctx)),
    _thread(move(other._thread)),
    _task_queue(move(other._task_queue)),
    _result_queue(move(other._result_queue))
{
    // Can't move the atmoics
    _running.store(other._running.load());
    _stop_requested.store(other._stop_requested.load());
    _completed_task_count.store(other._completed_task_count.load());
    _total_queued_pixels.store(other._total_queued_pixels.load());
    _pixels_completed.store(other._pixels_completed.load());
}

// Returns true if the thread was successfuly created and starts running
bool RenderThread::run() NOEXCEPT {
    try {
        // Create the thread object
        _thread = thread(&RenderThread::_thread_main_loop, this);
    } catch (...) {
        return false;
    }

    return true;
}

bool RenderThread::add_task(const Task &task) NOEXCEPT {
    try {
        {
            // Put the task on
            lock_guard guard(_task_queue_mutex);
            _task_queue.push(task);
        }

        // Store how many pixels need to be completed in total
        _total_queued_pixels.store(_total_queued_pixels.load() + _r_ctx.width);

        return true;
    } catch (...) {
        return false;
    }
}

bool RenderThread::retrive_result(Result &result) NOEXCEPT {
    lock_guard guard(_result_queue_mutex);

    // Nothing?
    if (_result_queue.empty())
        return false;

    // Else, got something, grab it and place it
    result = _result_queue.front();
    _result_queue.pop();
    return true;
}

void RenderThread::_thread_main_loop() NOEXCEPT {
    _running.store(true);
    _stop_requested.store(false);

    // Deep copy the scene, that way each thread has it's own local version to use
    if (_r_ctx.deep_copy_per_thread) {
        _r_ctx.camera = _r_ctx.camera->deep_copy();
        _r_ctx.scene = _r_ctx.scene->deep_copy();

        if (_r_ctx.lights)
            _r_ctx.lights = _r_ctx.lights->deep_copy();
    }

    Task task;
    bool got_task = false;

    while (_stop_requested.load() != true) {
        try {
            // Do render here

            got_task = _get_next_task(task);
            if (got_task)
                _store_result(task.scanline, _render_scanline(task));
            else
                this_thread::sleep_for(chrono::milliseconds(10));   // Take a short nap

        } catch (...) {
            // No errors leave thread
        }
    }

    _running.store(false);
}

vector<ColorRGBA> RenderThread::_render_scanline(const Task &task) NOEXCEPT {
    const auto scanline_num = static_cast<rreal>(task.scanline);
    vector<ColorRGBA> scanline(static_cast<size_t>(_r_ctx.width));
    RandomGenerator rng(task.rng_seed);

    for (uint16_t x = 0; x < _r_ctx.width; ++x) {
        // Put the pixel into the scanline
        scanline[x] = _pixel_color(_r_ctx, rng, task.samples_per_pixel, x, scanline_num, task.max_ray_depth);

        // Metering
        _pixels_completed.store(_pixels_completed.load() + 1);
    }

    return scanline;
}

bool RenderThread::_get_next_task(Task &task) NOEXCEPT {
    lock_guard guard(_task_queue_mutex);

    // Nothing?
    if (_task_queue.empty())
        return false;

    // There's something, grab it and place it
    task = _task_queue.front();
    _task_queue.pop();
    return true;
}

bool RenderThread::_store_result(const uint16_t scanline, const std::vector<ColorRGBA> &data) NOEXCEPT {
    try {
        const uint16_t num_completed_now = _completed_task_count.load() + 1;
        Result rr;
        rr.scanline = scanline;
        rr.data = data;

        // Put it onto the result queue (and notify)
        lock_guard guard(_result_queue_mutex);
        _result_queue.push(rr);
        _completed_task_count.store(num_completed_now);

        return true;
    } catch (...) {
        return false;
    }
}



Vec3 ray_color(const RenderContext &r_ctx, RandomGenerator &rng, const Ray &r, const uint32_t depth) NOEXCEPT {
    const RenderMethod r_method = r_ctx.render_method;

    // If we've exceeded the ray bounce limit, no more light is gathered
    if (depth <= 0)
        return Vec3(0);

    HitRecord h_rec{};

    // Anything?
    const bool hit_something = r_ctx.scene->hit(rng, r, static_cast<rreal>(0.001), Infinity, h_rec);
    if (!hit_something)
        return r_ctx.background;

    // Hit something...
    ScatterRecord s_rec;
    const Vec3 emitted = h_rec.mat_ptr->emitted(r, h_rec, h_rec.u, h_rec.v, h_rec.p, r_method);
    const bool rays_scattered = h_rec.mat_ptr->scatter(rng, r, h_rec, s_rec, r_method);

    // If hit a an emmsive material (they don't scatter rays), then only emit that one's light
    if (!rays_scattered)
        return emitted;

    // Chose a render method (for a non-emissive material)
    if (r_method == RenderMethod::Books1And2)
    {
        return emitted + (s_rec.attenuation * ray_color(r_ctx, rng, s_rec.ray, depth - 1));
    }
    else
    {
        if (s_rec.is_specular)
            return s_rec.attenuation * ray_color(r_ctx, rng, s_rec.ray, depth - 1);

        #ifdef USE_BOOK_PDF_MANAGEMENT
            const auto light_pdf = make_shared<HittablePDF>(r_ctx.lights, h_rec.p);
            const MixturePDF mixed_pdf(light_pdf, s_rec.pdf_ptr);
        #else
            // NOTE: The `s_rec.pdf` will always be of the type of CosinePDF because of what we put in our various
            //       IMaterial::scatter() methods has always been a CosinePDF.  The only reason I'm having that
            //       conversion function (`maybe_get_pdf_ptr()`) is for future flexabiliy.
            const HittablePDF light_pdf(r_ctx.lights, h_rec.p);
            const MixturePDF mixed_pdf(&light_pdf, maybe_get_pdf_ptr(s_rec.pdf));
        #endif  // USE_BOOK_PDF_MANAGEMENT

        const Ray scattered(h_rec.p, mixed_pdf.generate(rng), r.time);
        const rreal pdf_val = mixed_pdf.value(rng, scattered.direction);

        // else, it must have been non-emissive,
        return emitted + (s_rec.attenuation *
                          h_rec.mat_ptr->scattering_pdf(r, h_rec, scattered) *
                          ray_color(r_ctx, rng, scattered, depth - 1) / pdf_val);
    }
}


/**
 * Utility function to take PDVariant over to a pointer to the PDF interface.  If it wasn't
 * able to properly convert it, then it will return a nullptr.
 */
const IPDF *maybe_get_pdf_ptr(const PDFVariant &pdf) NOEXCEPT
{
    auto *ptr = static_cast<const IPDF *>(get_if<CosinePDF>(&pdf));
    if (ptr)
        return ptr;

    ptr = static_cast<const IPDF *>(get_if<HittablePDF>(&pdf));
    if (ptr)
        return ptr;

    // This is the final option (will either return a Mixture PDF or nullptr)
    ptr = static_cast<const IPDF *>(get_if<MixturePDF>(&pdf));
    return ptr;
}



ColorRGBA _pixel_color(const RenderContext &r_ctx, RandomGenerator &rng, const uint32_t samples_per_pixel, const rreal x, const rreal y, const uint16_t max_ray_depth) NOEXCEPT {
    // The result
    ColorRGBA pixel;

    const auto height_max = static_cast<uint16_t>(r_ctx.height - 1);
    const auto r_spp = static_cast<rreal>(samples_per_pixel);

#ifdef USE_BOOK_COMPUTE_PIXEL_COLOR
    for (uint32_t s = 0; s < samples_per_pixel; ++s) {
        const Ray r = r_ctx.camera->get_ray(
            rng,
                 (x + rng.get_real()) / static_cast<rreal>(r_ctx.width - 1),
            1 - ((y + rng.get_real()) / static_cast<rreal>(height_max))
        );
        const Vec3 clr = ray_color(r_ctx, rng, r, max_ray_depth);

        pixel.r += clr.x;
        pixel.g += clr.y;
        pixel.b += clr.z;
    }
#else
    // Compared to the book, this method instead involves pooling the randoms offsets, rays,
    // and colours and then doing the averaging after all the ray colors have been computed
    // Pool random offsets
    vector<array<rreal, 2>> uv_pairs(samples_per_pixel);
    for (auto &uv : uv_pairs) {
        uv[0] =      (x + rng.get_real()) / static_cast<rreal>(r_ctx.width - 1);
        uv[1] = 1 - ((y + rng.get_real()) / static_cast<rreal>(height_max));
    }

    // Pool some rays here
    vector<Ray> rays(samples_per_pixel);
    for (uint32_t s = 0; s < samples_per_pixel; ++s) {
        rays[s] = r_ctx.camera->get_ray(rng, uv_pairs[s][0], uv_pairs[s][1]);
    }

    vector<Vec3> clrs(samples_per_pixel);
    for (uint32_t s = 0; s < samples_per_pixel; ++s) {
        clrs[s] = ray_color(r_ctx, rng, rays[s], max_ray_depth);
    }

    // Put the pixel in their accumulator
    for (const Vec3 &c : clrs) {
        pixel.r += c.x;
        pixel.g += c.y;
        pixel.b += c.z;
    }
#endif  // USE_BOOK_COMPUTE_PIXEL_COLOR


    // Replace NaN components with zero (See explanation in the final chapter of "Ray Tracing: The Rest of Your Life")
    //   Note that this is not found in books 1 & 2, but having this for all shouldn't cause any harm.
#ifdef USE_BOOK_COMPUTE_PIXEL_COLOR
    if (pixel.r != pixel.r)
        pixel.r = 0;
    if (pixel.g != pixel.g)
        pixel.g = 0;
    if (pixel.b != pixel.b)
        pixel.b = 0;
//    if (pixel.a != pixel.a)
//        pixel.a = 0;
#else
    pixel.r = (pixel.r != pixel.r) ? 0 : pixel.r;
    pixel.g = (pixel.g != pixel.g) ? 0 : pixel.g;
    pixel.b = (pixel.b != pixel.b) ? 0 : pixel.b;
//    pixel.a = (pixel.a != pixel.a) ? 0 : pixel.a;
#endif  // USE_BOOK_COMPUTE_PIXEL_COLOR

    // Average all the samples
    pixel.r /= r_spp;
    pixel.g /= r_spp;
    pixel.b /= r_spp;
//    pixel.a /= r_spp;

    // Gamma correction (g=2.0)
    pixel.r = util::sqrt(pixel.r);
    pixel.g = util::sqrt(pixel.g);
    pixel.b = util::sqrt(pixel.b);
//    pixel.a = util::sqrt(pixel.a);

    return pixel;
}
